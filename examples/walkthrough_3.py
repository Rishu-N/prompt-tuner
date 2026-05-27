"""
Walkthrough Example 3 — Scientific Abstract TL;DR (length/concision)
=====================================================================
Scenario: The model is asked to produce a one-line TL;DR for a scientific
paper abstract. The initial mutable section just says "summarize the
abstract for a general audience" — it places no length bound, so the model
returns multi-sentence paragraphs. The optimizer should discover that it
needs to impose strict length bounds: a single sentence, between roughly
15 and 30 words, no semicolons, no parenthetical asides, and no in-line
citations.

Run:
  python examples/walkthrough_3.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from core.models import ModelConfig, ModelClient
from core.prompt import Prompt, PromptSection
from core.test_case import TestCase
from core.runner import run_optimization

API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = "https://api.openai.com/v1"
TARGET_MODEL = "gpt-4o-mini"
SUPERVISOR_MODEL = "gpt-4o-mini"

prompt = Prompt([
    PromptSection(
        text=(
            "You are a science communicator. The user will paste a scientific "
            "paper abstract and you will produce a TL;DR that helps a general "
            "reader decide whether to read the paper."
        ),
        mutable=False,
    ),
    PromptSection(
        text="Summarize the abstract for a general audience.",
        mutable=True,
    ),
]
)

test_cases = [
    TestCase(
        name="crispr_abstract",
        input_text=(
            "Background: CRISPR-Cas9 has revolutionized genome editing, but "
            "off-target effects remain a clinical concern. Methods: We engineered "
            "a high-fidelity Cas9 variant (HF-Cas9-v2) and benchmarked it against "
            "wild-type Cas9 across 24 genomic loci in primary human T cells using "
            "GUIDE-seq. Results: HF-Cas9-v2 reduced off-target edits by 87% on "
            "average (p < 0.001) while preserving on-target activity within 5% of "
            "wild-type. Conclusion: HF-Cas9-v2 is a promising tool for "
            "therapeutic genome editing where specificity is paramount."
        ),
        expected_output=(
            "A single sentence, between 15 and 30 words, conveying that a new "
            "high-fidelity CRISPR-Cas9 variant sharply reduces off-target edits "
            "while keeping on-target activity, making it suitable for therapeutic "
            "use. The output MUST be exactly one sentence (one period or "
            "equivalent terminator), MUST NOT contain semicolons, MUST NOT "
            "contain parentheses, and MUST NOT contain any numerical citation "
            "like (Smith 2020) or [12]."
        ),
    ),
    TestCase(
        name="exoplanet_abstract",
        input_text=(
            "We report the discovery of TOI-4823b, a sub-Neptune exoplanet "
            "transiting a nearby K-dwarf star at 38 parsecs. Using TESS photometry "
            "and HARPS radial velocities, we measure a radius of 2.1 R_Earth and a "
            "mass of 6.4 M_Earth, implying a bulk density consistent with a "
            "volatile-rich interior. The planet's equilibrium temperature of 420 K "
            "and bright host star (V=9.1) make it an attractive target for "
            "atmospheric characterization with JWST."
        ),
        expected_output=(
            "A single sentence, 15 to 30 words, stating that astronomers found "
            "a sub-Neptune planet around a nearby K-dwarf with properties making "
            "it a strong candidate for JWST atmospheric study. MUST be exactly "
            "one sentence, MUST NOT contain semicolons or parentheses, and MUST "
            "NOT contain citation markers."
        ),
    ),
]

target_cfg     = ModelConfig(model_id=TARGET_MODEL,     api_key=API_KEY, base_url=BASE_URL)
supervisor_cfg = ModelConfig(model_id=SUPERVISOR_MODEL, api_key=API_KEY, base_url=BASE_URL)
target     = ModelClient(target_cfg)
supervisor = ModelClient(supervisor_cfg)

print("=" * 60)
print("INITIAL PROMPT")
print("=" * 60)
print(prompt.render_annotated())
print()

def log(msg): print(msg)

history = run_optimization(
    prompt=prompt,
    test_cases=test_cases,
    target=target,
    supervisor=supervisor,
    epochs=3,
    max_iterations=3,
    log_callback=log,
)

print()
print("=" * 60)
print("FINAL PROMPT")
print("=" * 60)
print(history.final_prompt.render_annotated())
print()
print("Pass rates per epoch:", [f"{r:.0%}" for r in history.pass_rates()])
