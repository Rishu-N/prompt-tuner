"""
Prompt Optimizer — Gradio UI
Run with:  python app.py
"""
import threading
import difflib
import json
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from config import load_feature_flags
from core.models import ModelConfig, ModelClient, build_client
from core.prompt import Prompt, PromptSection
from core.test_case import TestCase
from core.runner import run_optimization, OptimizationHistory
from reliability.cost_tracker import CostTracker
from utils.serialization import save_session, load_session
from utils.export import export_results_csv, export_results_json, export_final_prompt

FLAGS = load_feature_flags()

# -----------------------------------------------------------------------
# Thread-safe state (3A)
# -----------------------------------------------------------------------

class ThreadSafeState:
    def __init__(self):
        self._lock = threading.Lock()
        self._history: OptimizationHistory | None = None
        self._stop_flag: bool = False
        self._log_lines: list[str] = []
        self._cost_tracker: CostTracker | None = None

    def log(self, msg: str) -> None:
        with self._lock:
            self._log_lines.append(msg)

    def get_log(self) -> str:
        with self._lock:
            return "\n".join(self._log_lines)

    @property
    def history(self) -> OptimizationHistory | None:
        with self._lock:
            return self._history

    @history.setter
    def history(self, value):
        with self._lock:
            self._history = value

    @property
    def stop_flag(self) -> bool:
        with self._lock:
            return self._stop_flag

    @stop_flag.setter
    def stop_flag(self, value: bool):
        with self._lock:
            self._stop_flag = value

    @property
    def cost_tracker(self) -> CostTracker | None:
        with self._lock:
            return self._cost_tracker

    @cost_tracker.setter
    def cost_tracker(self, value):
        with self._lock:
            self._cost_tracker = value

    def reset(self):
        with self._lock:
            self._history = None
            self._stop_flag = False
            self._log_lines = []
            self._cost_tracker = CostTracker() if FLAGS.cost_tracking_enabled else None


_state = ThreadSafeState()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _build_prompt(line_table: list[list]) -> Prompt:
    if not line_table:
        return Prompt([PromptSection(text="", mutable=True)])
    lines = [row[0] for row in line_table]
    flags = [bool(row[1]) for row in line_table]
    return Prompt.from_lines_with_mutability(lines, flags)


def _prompt_to_table(prompt: Prompt) -> list[list]:
    lines, flags = prompt.to_lines_with_mutability()
    return [[line, flag] for line, flag in zip(lines, flags)]


def _build_test_cases(tc_table: list[list]) -> list[TestCase]:
    cases = []
    for row in tc_table:
        name = str(row[0]).strip()
        inp = str(row[1]).strip()
        exp = str(row[2]).strip()
        if inp:
            cases.append(TestCase(name=name or None, input_text=inp, expected_output=exp))
    return cases


def _build_model_config(base_url: str, model_id: str, api_key: str, name: str = "") -> ModelConfig:
    return ModelConfig(
        model_id=model_id.strip(),
        api_key=api_key.strip(),
        base_url=base_url.strip() or None,
        name=name or model_id.strip(),
    )


# -----------------------------------------------------------------------
# Prompt editor
# -----------------------------------------------------------------------

def load_prompt_text(raw_text: str):
    lines = raw_text.split("\n")
    table = [[line, True] for line in lines]
    preview = _render_color_preview(table)
    return table, preview


def mark_lines_immutable(table: list[list], indices_str: str):
    """Mark comma-separated line numbers (1-based) as immutable."""
    try:
        indices = [int(x.strip()) - 1 for x in indices_str.split(",") if x.strip()]
    except ValueError:
        return table, _render_color_preview(table)
    for i in indices:
        if 0 <= i < len(table):
            table[i][1] = False
    return table, _render_color_preview(table)


def update_color_preview(table: list[list]):
    return _render_color_preview(table)


def _render_color_preview(table: list[list]) -> str:
    if not FLAGS.ui_color_preview or not table:
        return ""
    html_lines = []
    for idx, row in enumerate(table):
        text = row[0] if row[0] else "&nbsp;"
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        color = "#e8f5e9" if row[1] else "#f5f5f5"
        label = "M" if row[1] else "I"
        html_lines.append(
            f'<div style="background:{color};padding:2px 6px;margin:1px 0;'
            f'font-family:monospace;font-size:13px;border-left:3px solid '
            f'{"#4caf50" if row[1] else "#9e9e9e"}">'
            f'<span style="color:#888;font-size:11px">{idx+1} [{label}]</span> {text}</div>'
        )
    return "".join(html_lines)


# -----------------------------------------------------------------------
# Input validation (3C)
# -----------------------------------------------------------------------

def validate_inputs(line_table, tc_table, sup_api_key, tgt_api_key, sup_model_id, tgt_model_id):
    if not FLAGS.ui_input_validation:
        return None
    errors = []
    if not line_table or all(not str(row[0]).strip() for row in line_table):
        errors.append("Prompt is empty.")
    if line_table and not any(row[1] for row in line_table):
        errors.append("No mutable lines — optimizer cannot change anything.")
    if not tc_table or not any(str(row[1]).strip() for row in tc_table):
        errors.append("No test cases with input text.")
    if not sup_api_key.strip():
        errors.append("Supervisor API key is missing.")
    if not tgt_api_key.strip():
        errors.append("Target API key is missing.")
    if not sup_model_id.strip():
        errors.append("Supervisor model ID is missing.")
    if not tgt_model_id.strip():
        errors.append("Target model ID is missing.")
    return "; ".join(errors) if errors else None


# -----------------------------------------------------------------------
# Run tab
# -----------------------------------------------------------------------

def start_run(
    line_table, tc_table,
    sup_base_url, sup_model_id, sup_api_key,
    tgt_base_url, tgt_model_id, tgt_api_key,
    epochs, max_iters,
):
    # Validation (3C)
    err = validate_inputs(line_table, tc_table, sup_api_key, tgt_api_key, sup_model_id, tgt_model_id)
    if err:
        return f"Validation failed: {err}", ""

    prompt = _build_prompt(line_table)
    test_cases = _build_test_cases(tc_table)
    if not test_cases:
        return "No test cases defined.", ""

    supervisor_cfg = _build_model_config(sup_base_url, sup_model_id, sup_api_key, "Supervisor")
    target_cfg = _build_model_config(tgt_base_url, tgt_model_id, tgt_api_key, "Target")

    _state.reset()

    try:
        supervisor = build_client(supervisor_cfg, FLAGS, _state.cost_tracker)
        target = build_client(target_cfg, FLAGS, _state.cost_tracker)
    except Exception as e:
        return f"Error initialising models: {e}", ""

    def _run():
        try:
            history = run_optimization(
                prompt=prompt,
                test_cases=test_cases,
                target=target,
                supervisor=supervisor,
                epochs=int(epochs),
                max_iterations=int(max_iters),
                log_callback=_state.log,
                stop_flag=lambda: _state.stop_flag,
                flags=FLAGS,
            )
            _state.history = history
            # Cost summary
            if FLAGS.cost_tracking_enabled and _state.cost_tracker:
                summary = _state.cost_tracker.summary()
                _state.log(
                    f"\nCost: ${summary['total_cost_usd']:.4f} | "
                    f"Tokens: {summary['total_tokens']:,} | "
                    f"Calls: {summary['call_count']}"
                )
            _state.log("=== Run complete ===")
        except Exception as e:
            _state.log(f"ERROR: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return "Run started — check the log below.", ""


def stop_run():
    _state.stop_flag = True
    return "Stop requested."


def poll_log():
    return _state.get_log()


# -----------------------------------------------------------------------
# Results tab
# -----------------------------------------------------------------------

def get_epoch_choices():
    history = _state.history
    if not history or not history.epoch_results:
        return gr.Dropdown(choices=[], value=None)
    choices = [f"Epoch {r.epoch}" for r in history.epoch_results]
    return gr.Dropdown(choices=choices, value=choices[-1])


def build_pass_rate_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = _state.history
    if not history or not history.epoch_results:
        fig, ax = plt.subplots()
        ax.set_title("No data yet")
        return fig

    rates = [r.pass_rate * 100 for r in history.epoch_results]
    epochs = [r.epoch for r in history.epoch_results]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(epochs, rates, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Prompt Optimization Progress")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def get_epoch_details(epoch_label: str):
    history = _state.history
    if not history or not epoch_label:
        return "", [], ""

    epoch_num = int(epoch_label.split()[-1])
    result = next((r for r in history.epoch_results if r.epoch == epoch_num), None)
    if not result:
        return "", [], ""

    before_lines = result.prompt_before.render().splitlines(keepends=True)
    after_lines = result.prompt_after.render().splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(before_lines, after_lines, fromfile="before", tofile="after"))
    diff_text = diff if diff else "(no changes this epoch)"

    rows = []
    for tr in result.test_results:
        rows.append([
            tr.test_name,
            "PASS" if tr.passed else "FAIL",
            tr.eval_result.feedback[:120],
            tr.actual_output[:120],
        ])

    final_prompt = result.prompt_after.render()
    return diff_text, rows, final_prompt


# -----------------------------------------------------------------------
# Export (3D)
# -----------------------------------------------------------------------

def export_csv():
    history = _state.history
    if not history:
        return None
    content = export_results_csv(history)
    path = os.path.join(tempfile.gettempdir(), "prompt_optimizer_results.csv")
    Path(path).write_text(content)
    return path


def export_json():
    history = _state.history
    if not history:
        return None
    content = export_results_json(history)
    path = os.path.join(tempfile.gettempdir(), "prompt_optimizer_results.json")
    Path(path).write_text(content)
    return path


# -----------------------------------------------------------------------
# Session save/load
# -----------------------------------------------------------------------

def save_session_handler(line_table, tc_table, sup_base_url, sup_model_id, sup_api_key,
                         tgt_base_url, tgt_model_id, tgt_api_key, epochs, max_iters, filepath):
    try:
        prompt = _build_prompt(line_table)
        test_cases = _build_test_cases(tc_table)
        sup_cfg = _build_model_config(sup_base_url, sup_model_id, sup_api_key)
        tgt_cfg = _build_model_config(tgt_base_url, tgt_model_id, tgt_api_key)
        save_session(filepath, prompt, test_cases, tgt_cfg, sup_cfg, int(epochs), int(max_iters))
        return f"Saved to {filepath}"
    except Exception as e:
        return f"Error: {e}"


def load_session_handler(filepath):
    try:
        session = load_session(filepath)
        prompt: Prompt = session["prompt"]
        line_table = _prompt_to_table(prompt)
        tc_table = [[tc.name, tc.input_text, tc.expected_output] for tc in session["test_cases"]]
        tgt = session["target_config"]
        sup = session["supervisor_config"]
        return (
            line_table, tc_table,
            sup.base_url or "", sup.model_id, sup.api_key,
            tgt.base_url or "", tgt.model_id, tgt.api_key,
            session["epochs"], session["max_iterations"],
            f"Loaded from {filepath}",
        )
    except Exception as e:
        return [None] * 10 + [f"Error: {e}"]


# -----------------------------------------------------------------------
# Cost dashboard (3F)
# -----------------------------------------------------------------------

def get_cost_summary():
    if not FLAGS.cost_tracking_enabled or not _state.cost_tracker:
        return "Cost tracking is disabled. Enable via config.yaml: cost_tracking_enabled: true"
    return json.dumps(_state.cost_tracker.summary(), indent=2)


# -----------------------------------------------------------------------
# UI layout
# -----------------------------------------------------------------------

PROMPT_LINE_HEADERS = ["Line Text", "Mutable?"]
TC_HEADERS = ["Name", "Input", "Expected Output"]
RESULT_HEADERS = ["Test Name", "Result", "Feedback", "Actual Output (truncated)"]

with gr.Blocks(title="Prompt Optimizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Prompt Optimizer\nIteratively improve your prompts using an LLM supervisor loop.")

    with gr.Tabs():
        # ----------------------------------------------------------------
        # Tab 1: Prompt Editor
        # ----------------------------------------------------------------
        with gr.Tab("1 · Prompt Editor"):
            gr.Markdown(
                "Enter your prompt below, then mark individual lines as **Mutable** (can be changed) "
                "or **Immutable** (locked). The optimizer will only modify mutable lines."
            )
            raw_prompt_input = gr.Textbox(
                label="Paste prompt here to load into the line editor",
                placeholder="Enter your full prompt here...",
                lines=6,
            )
            load_prompt_btn = gr.Button("Load into Line Editor", variant="secondary")
            prompt_table = gr.Dataframe(
                headers=PROMPT_LINE_HEADERS,
                datatype=["str", "bool"],
                col_count=(2, "fixed"),
                label="Prompt Lines (check Mutable = optimizer can change this line)",
                interactive=True,
                wrap=True,
            )
            # 3G: Wire mark_selected_immutable
            with gr.Row():
                immutable_indices = gr.Textbox(
                    label="Mark lines immutable (comma-separated line numbers, 1-based)",
                    placeholder="e.g. 1, 2, 5",
                    scale=3,
                )
                mark_immutable_btn = gr.Button("Mark Immutable", variant="secondary", scale=1)

            # 3E: Color-coded preview
            color_preview = gr.HTML(label="Prompt Preview", visible=FLAGS.ui_color_preview)

            load_prompt_btn.click(
                fn=load_prompt_text,
                inputs=[raw_prompt_input],
                outputs=[prompt_table, color_preview],
            )
            mark_immutable_btn.click(
                fn=mark_lines_immutable,
                inputs=[prompt_table, immutable_indices],
                outputs=[prompt_table, color_preview],
            )
            if FLAGS.ui_color_preview:
                prompt_table.change(
                    fn=update_color_preview,
                    inputs=[prompt_table],
                    outputs=[color_preview],
                )

        # ----------------------------------------------------------------
        # Tab 2: Test Cases
        # ----------------------------------------------------------------
        with gr.Tab("2 · Test Cases"):
            gr.Markdown("Add test cases. Each row is one test: a user input and the expected model output.")
            tc_table = gr.Dataframe(
                headers=TC_HEADERS,
                datatype=["str", "str", "str"],
                col_count=(3, "fixed"),
                label="Test Cases",
                interactive=True,
                wrap=True,
                value=[["test_1", "", ""]],
            )

        # ----------------------------------------------------------------
        # Tab 3: Model Config
        # ----------------------------------------------------------------
        with gr.Tab("3 · Model Config"):
            gr.Markdown(
                "Configure both models. Use any OpenAI-compatible endpoint: Anthropic, OpenRouter, "
                "Groq, Together, local Ollama, etc."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Supervisor Model")
                    gr.Markdown("*Evaluates outputs and reviews proposed prompt changes.*")
                    sup_base_url = gr.Textbox(label="Base URL (leave blank for OpenAI default)", placeholder="https://api.anthropic.com/v1")
                    sup_model_id = gr.Textbox(label="Model ID", placeholder="claude-sonnet-4-6")
                    sup_api_key = gr.Textbox(label="API Key", type="password")
                with gr.Column():
                    gr.Markdown("### Target Model")
                    gr.Markdown("*The model whose prompt is being optimized.*")
                    tgt_base_url = gr.Textbox(label="Base URL (leave blank for OpenAI default)", placeholder="https://api.openai.com/v1")
                    tgt_model_id = gr.Textbox(label="Model ID", placeholder="gpt-4o")
                    tgt_api_key = gr.Textbox(label="API Key", type="password")

        # ----------------------------------------------------------------
        # Tab 4: Run
        # ----------------------------------------------------------------
        with gr.Tab("4 · Run"):
            with gr.Row():
                epochs_input = gr.Number(label="Epochs", value=3, minimum=1, maximum=50, step=1)
                max_iters_input = gr.Slider(label="Max optimizer iterations per failure", minimum=1, maximum=5, value=3, step=1)
            with gr.Row():
                run_btn = gr.Button("Run Optimization", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")
            run_status = gr.Textbox(label="Status", interactive=False)
            log_output = gr.Textbox(label="Live Log", lines=20, interactive=False, autoscroll=True)
            poll_btn = gr.Button("Refresh Log", variant="secondary")

            run_btn.click(
                fn=start_run,
                inputs=[
                    prompt_table, tc_table,
                    sup_base_url, sup_model_id, sup_api_key,
                    tgt_base_url, tgt_model_id, tgt_api_key,
                    epochs_input, max_iters_input,
                ],
                outputs=[run_status, log_output],
            )
            stop_btn.click(fn=stop_run, outputs=[run_status])
            poll_btn.click(fn=poll_log, outputs=[log_output])

        # ----------------------------------------------------------------
        # Tab 5: Results
        # ----------------------------------------------------------------
        with gr.Tab("5 · Results"):
            refresh_results_btn = gr.Button("Refresh Results", variant="secondary")
            pass_rate_plot = gr.Plot(label="Pass Rate per Epoch")
            epoch_selector = gr.Dropdown(label="Select Epoch to inspect", choices=[], interactive=True)
            with gr.Row():
                diff_output = gr.Textbox(label="Prompt Diff (this epoch)", lines=15, interactive=False)
                final_prompt_output = gr.Textbox(label="Prompt after epoch", lines=15, interactive=False)
            test_result_table = gr.Dataframe(
                headers=RESULT_HEADERS,
                datatype=["str", "str", "str", "str"],
                col_count=(4, "fixed"),
                label="Per-test Results",
                wrap=True,
            )

            # 3D: Export buttons
            if FLAGS.ui_export_enabled:
                with gr.Row():
                    export_csv_btn = gr.Button("Export CSV")
                    export_json_btn = gr.Button("Export JSON")
                export_file = gr.File(label="Download", visible=FLAGS.ui_export_enabled)
                export_csv_btn.click(fn=export_csv, outputs=[export_file])
                export_json_btn.click(fn=export_json, outputs=[export_file])

            def refresh_results():
                chart = build_pass_rate_chart()
                dd = get_epoch_choices()
                return chart, dd

            refresh_results_btn.click(
                fn=refresh_results,
                outputs=[pass_rate_plot, epoch_selector],
            )
            epoch_selector.change(
                fn=get_epoch_details,
                inputs=[epoch_selector],
                outputs=[diff_output, test_result_table, final_prompt_output],
            )

        # ----------------------------------------------------------------
        # Tab 6: Session
        # ----------------------------------------------------------------
        with gr.Tab("6 · Session"):
            gr.Markdown("Save or load your full configuration (prompt, test cases, model settings).")
            session_filepath = gr.Textbox(label="File path", value="session.json")
            with gr.Row():
                save_btn = gr.Button("Save Session")
                load_btn = gr.Button("Load Session")
            session_status = gr.Textbox(label="Status", interactive=False)

            save_btn.click(
                fn=save_session_handler,
                inputs=[
                    prompt_table, tc_table,
                    sup_base_url, sup_model_id, sup_api_key,
                    tgt_base_url, tgt_model_id, tgt_api_key,
                    epochs_input, max_iters_input,
                    session_filepath,
                ],
                outputs=[session_status],
            )
            load_btn.click(
                fn=load_session_handler,
                inputs=[session_filepath],
                outputs=[
                    prompt_table, tc_table,
                    sup_base_url, sup_model_id, sup_api_key,
                    tgt_base_url, tgt_model_id, tgt_api_key,
                    epochs_input, max_iters_input,
                    session_status,
                ],
            )

        # ----------------------------------------------------------------
        # Tab 7: Cost Dashboard (3F) — only visible when enabled
        # ----------------------------------------------------------------
        if FLAGS.ui_cost_dashboard:
            with gr.Tab("7 · Cost Dashboard"):
                gr.Markdown("### Token Usage & Cost Tracking")
                cost_refresh_btn = gr.Button("Refresh", variant="secondary")
                cost_output = gr.Code(label="Cost Summary (JSON)", language="json", lines=10)
                cost_refresh_btn.click(fn=get_cost_summary, outputs=[cost_output])


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
