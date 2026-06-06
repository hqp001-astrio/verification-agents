"""Compare base LLM (natural-language judgment) vs the Z3-grounded verifier.

For each labeled case:
  * base    -> an LLM judges yes/no/unsure from the code in natural language.
  * verifier-> the specialist+generalist pipeline (Z3 + execution) decides, and we
              record whether the verdict is GROUNDED (backed by a Z3 UNSAT proof or
              an execution-reproduced counterexample).

We report accuracy plus the metrics that show the advantage: missed-bug rate
(false negatives — the dangerous ones) and false-alarm rate (false positives on
safe code), and how many verdicts were grounded vs asserted. Run:
    uv run python -m verification_agents.eval.run_eval
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from verification_agents.eval.dataset import CASES, Case
from verification_agents.specialists.verify import verify
from verification_agents.tools import parse_diff as _parse_diff

load_dotenv()
_MODEL = os.environ.get("VERIFY_MODEL", "gpt-4o-mini")        # the verifier's encoder
_BASE_MODEL = os.environ.get("BASE_MODEL", _MODEL)            # the LLM under comparison


def _is_reasoning(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _chat(client, model: str, system: str, user: str) -> str:
    """Model-aware chat call (reasoning models reject temperature/max_tokens)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if _is_reasoning(model):
        resp = client.chat.completions.create(model=model, messages=msgs,
                                              max_completion_tokens=4096)
    else:
        resp = client.chat.completions.create(model=model, messages=msgs,
                                              temperature=0, max_tokens=8)
    return resp.choices[0].message.content or ""


def _chat_anthropic(model: str, system: str, user: str) -> str:
    """One-shot classification via the Anthropic SDK. Opus 4.8 rejects temperature
    and returns content as a list of blocks."""
    import anthropic

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from the env
    resp = client.messages.create(
        model=model, max_tokens=256, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")

try:
    import weave

    _HAS_WEAVE = True
except Exception:
    weave = None
    _HAS_WEAVE = False


def _op(name):
    def deco(fn):
        return weave.op(fn, name=name) if _HAS_WEAVE else fn
    return deco


_BASE_SYSTEM = (
    "You are a code reviewer. Judge the function over ALL inputs of its parameter "
    "types for safety bugs (array out-of-bounds, null/None dereference, division by "
    "zero, non-termination). Reply with EXACTLY ONE word:\n"
    "  BUG    - some valid-typed input crashes or violates safety\n"
    "  SAFE   - no valid-typed input does\n"
    "  UNSURE - safety depends on state you cannot see (a global, a network/IO call)\n"
    "Output only that one word."
)

_BASE_MAP = {"bug": "no", "safe": "yes", "unsure": "unsure"}


def _code_to_diff(code: str, name: str) -> str:
    lines = code.splitlines() or [""]
    body = "\n".join("+" + ln for ln in lines)
    return (f"diff --git a/{name}.py b/{name}.py\nnew file mode 100644\n"
            f"--- /dev/null\n+++ b/{name}.py\n@@ -0,0 +1,{len(lines)} @@\n{body}\n")


@_op("eval.base_llm")
def base_llm(case: Case, api_key: str) -> str:
    prompt = f"```python\n{case.code}\n```"
    if _BASE_MODEL.startswith("claude"):
        word = _chat_anthropic(_BASE_MODEL, _BASE_SYSTEM, prompt).strip().lower()
    else:
        import openai

        client = openai.OpenAI(api_key=api_key)
        word = _chat(client, _BASE_MODEL, _BASE_SYSTEM, prompt).strip().lower()
    # match the most specific token first so "no bug" isn't read as a bug
    for token in ("unsure", "safe", "bug"):
        if token in word:
            return _BASE_MAP[token]
    return "unsure"


@_op("eval.verifier")
def verifier(case: Case, api_key: str) -> dict:
    analysis = _parse_diff.run(_code_to_diff(case.code, case.name))
    report = verify(analysis, api_key=api_key, model=_MODEL)
    # "grounded": the verdict is backed by a proof or a reproduction, not an opinion.
    grounded = (
        (report.decision == "no" and any(b.execution_confirmed for b in report.bugs))
        or (report.decision == "yes" and bool(report.clean))
    )
    return {"decision": report.decision, "grounded": bool(grounded)}


def _metrics(preds: list[str], golds: list[str]) -> dict:
    n = len(golds)
    acc = sum(p == g for p, g in zip(preds, golds)) / n
    bug_total = sum(g == "no" for g in golds)
    safe_total = sum(g == "yes" for g in golds)
    missed = sum(g == "no" and p != "no" for p, g in zip(preds, golds))   # false negatives
    false_alarm = sum(g == "yes" and p == "no" for p, g in zip(preds, golds))  # false positives
    return {
        "accuracy": acc,
        "missed_bug_rate": missed / bug_total if bug_total else 0.0,
        "false_alarm_rate": false_alarm / safe_total if safe_total else 0.0,
    }


@_op("eval.run")
def run(cases: list[Case] | None = None) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY required for the eval")
    cases = cases or CASES

    golds = [c.label for c in cases]
    base_preds, ver_preds, grounded = [], [], 0
    rows = []
    for case in cases:
        b = base_llm(case, api_key)
        v = verifier(case, api_key)
        base_preds.append(b)
        ver_preds.append(v["decision"])
        grounded += v["grounded"]
        rows.append((case.name, case.label, b, v["decision"], v["grounded"]))

    bm, vm = _metrics(base_preds, golds), _metrics(ver_preds, golds)
    n = len(cases)

    print(f"\nbase: {_BASE_MODEL}    verifier-encoder: {_MODEL}    dataset: {len(cases)} cases")
    print(f"{'case':22} {'gold':7} {'base':7} {'verifier':9} grounded")
    print("-" * 56)
    for name, gold, b, v, g in rows:
        flag = lambda p: "✓" if p == gold else "✗"  # noqa: E731
        print(f"{name:20} {gold:7} {b:1}{flag(b):>2}    {v:1}{flag(v):>2}      {'●' if g else '·'}")
    print("-" * 56)
    print(f"{'accuracy':22} base {bm['accuracy']:.0%}     verifier {vm['accuracy']:.0%}")
    print(f"{'missed-bug rate (FN)':22} base {bm['missed_bug_rate']:.0%}     verifier {vm['missed_bug_rate']:.0%}")
    print(f"{'false-alarm rate (FP)':22} base {bm['false_alarm_rate']:.0%}     verifier {vm['false_alarm_rate']:.0%}")
    print(f"{'grounded verdicts':22} base  0/{n}    verifier {grounded}/{n}  (proof or reproduction)")
    return {"n": n, "base": bm, "verifier": vm, "grounded": grounded}


if __name__ == "__main__":
    import sys

    if _HAS_WEAVE and os.environ.get("WANDB_API_KEY"):
        try:
            weave.init(os.environ.get("WEAVE_PROJECT", "astrio/verification-agents"))
        except Exception as exc:
            print(f"[eval] weave disabled: {exc}")

    chosen = CASES
    if "--hard" in sys.argv:
        from verification_agents.eval.dataset_hard import CASES_HARD

        chosen = CASES_HARD
    elif "--all" in sys.argv:
        from verification_agents.eval.dataset_hard import CASES_HARD

        chosen = CASES + CASES_HARD
    run(chosen)
