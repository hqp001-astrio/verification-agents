"""W&B Weave Evaluation: base LLM vs verifier as a side-by-side leaderboard.

This produces the judge-facing artifact in the Weave UI (two models, one dataset,
shared scorers). Requires WANDB_API_KEY. Run:
    uv run python -m verification_agents.eval.weave_eval
"""

from __future__ import annotations

import asyncio
import os

import weave
from dotenv import load_dotenv

from verification_agents.eval.dataset import CASES, Case
from verification_agents.eval.run_eval import base_llm, verifier

load_dotenv()
_KEY = os.environ.get("OPENAI_API_KEY")


class BaseLLMReviewer(weave.Model):
    @weave.op
    def predict(self, name: str, code: str) -> dict:
        return {"decision": base_llm(Case(name=name, code=code, label="?"), _KEY)}


class Z3Verifier(weave.Model):
    @weave.op
    def predict(self, name: str, code: str) -> dict:
        return verifier(Case(name=name, code=code, label="?"), _KEY)


@weave.op
def score(label: str, output: dict) -> dict:
    decision = output.get("decision")
    return {
        "correct": decision == label,
        "missed_bug": label == "no" and decision != "no",      # false negative
        "false_alarm": label == "yes" and decision == "no",    # false positive
        "grounded": bool(output.get("grounded", False)),
    }


def main() -> None:
    if not _KEY:
        raise SystemExit("OPENAI_API_KEY required")
    weave.init(os.environ.get("WEAVE_PROJECT", "astrio/verification-agents"))
    rows = [{"name": c.name, "code": c.code, "label": c.label} for c in CASES]
    evaluation = weave.Evaluation(dataset=rows, scorers=[score])
    print("=== base LLM ===")
    asyncio.run(evaluation.evaluate(BaseLLMReviewer()))
    print("=== Z3 verifier ===")
    asyncio.run(evaluation.evaluate(Z3Verifier()))
    print("Open the Weave UI to compare the two models side by side.")


if __name__ == "__main__":
    main()
