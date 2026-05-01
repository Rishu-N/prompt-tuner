"""Export optimization results as CSV or JSON."""
from __future__ import annotations

import csv
import json
import io
from pathlib import Path

from core.runner import OptimizationHistory


def export_results_csv(history: OptimizationHistory) -> str:
    """Return a CSV string of all test results across all epochs."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Epoch", "Test Name", "Passed", "Feedback", "Actual Output", "Expected Output"])
    for epoch_result in history.epoch_results:
        for tr in epoch_result.test_results:
            writer.writerow([
                epoch_result.epoch,
                tr.test_name,
                tr.passed,
                tr.eval_result.feedback,
                tr.actual_output,
                tr.expected_output,
            ])
    return output.getvalue()


def export_results_json(history: OptimizationHistory) -> str:
    """Return a JSON string of the full optimization history."""
    data = {
        "epochs": [
            {
                "epoch": er.epoch,
                "pass_rate": er.pass_rate,
                "pass_count": er.pass_count,
                "total_count": er.total_count,
                "duration_seconds": er.duration_seconds,
                "prompt_before": er.prompt_before.render(),
                "prompt_after": er.prompt_after.render(),
                "test_results": [
                    {
                        "name": tr.test_name,
                        "passed": tr.passed,
                        "feedback": tr.eval_result.feedback,
                        "reasoning": tr.eval_result.reasoning,
                        "actual_output": tr.actual_output,
                        "expected_output": tr.expected_output,
                    }
                    for tr in er.test_results
                ],
            }
            for er in history.epoch_results
        ],
        "final_prompt": history.final_prompt.render() if history.final_prompt else None,
        "pass_rates": history.pass_rates(),
    }
    return json.dumps(data, indent=2)


def export_final_prompt(history: OptimizationHistory) -> str:
    """Return the final prompt text."""
    if history.final_prompt:
        return history.final_prompt.render()
    return ""
