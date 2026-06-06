"""Run the specialist-column verification pipeline.

Usage:
    uv run python run_specialists.py demo            # all three toy examples
    uv run python run_specialists.py demo:bug        # one example (bug|safe|unsure)
    uv run python run_specialists.py path/to/file.diff
    uv run python run_specialists.py --repo ../app main HEAD   # diff a git range

Runs offline (heuristic encoder) with no OPENAI_API_KEY, or with the real
specialist LLM when the key is set. Streams stage events, shows the Redis backend,
and prints a Weave trace link when WANDB_API_KEY is configured.
"""

from __future__ import annotations

import os
import subprocess
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from verification_agents.specialists.examples import EXAMPLES
from verification_agents.specialists.verify import verify as verify_op
from verification_agents.tools import parse_diff as parse_diff_mod

load_dotenv()
console = Console()

_DECISION_STYLE = {"no": "bold red", "yes": "bold green", "unsure": "bold yellow"}


def _init_weave() -> bool:
    """Best-effort Weave init; gated on WANDB_API_KEY so headless runs never block."""
    if not os.environ.get("WANDB_API_KEY"):
        return False
    try:
        import weave

        weave.init(os.environ.get("WEAVE_PROJECT", "astrio/verification-agents"))
        return True
    except Exception as exc:
        console.print(f"[dim]weave disabled: {exc}[/dim]")
        return False


def _run(analysis, weave_on: bool):
    key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("VERIFY_MODEL", "gpt-4o-mini")
    kwargs = dict(api_key=key, model=model, z3_timeout_ms=10_000)

    # Capture the Weave trace URL via op.call(...) when tracing is active.
    if weave_on and hasattr(verify_op, "call"):
        report, call = verify_op.call(analysis, **kwargs)
        report.weave_trace_url = getattr(call, "ui_url", "") or ""
    else:
        report = verify_op(analysis, **kwargs)
    return report


def _print(report, title: str) -> None:
    style = _DECISION_STYLE.get(report.decision, "bold")
    console.print(Panel.fit(
        f"[{style}]{report.decision.upper()}[/{style}]  —  {title}",
        title="verification",
    ))

    cols = ", ".join(f"{c.label}×{c.n_constraints}" for c in report.columns_run) or "none"
    console.print(f"[dim]specialists:[/dim] {cols}   "
                  f"[dim]backend:[/dim] {report.backend}   "
                  f"[dim]checks:[/dim] {report.total_constraints}   "
                  f"[dim]{report.elapsed_s:.2f}s[/dim]")

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("verdict", width=12)
    table.add_column("concern", width=18)
    table.add_column("location", width=26)
    table.add_column("detail")
    for o in report.bugs:
        ce = ", ".join(f"{k}={v}" for k, v in (o.counterexample or {}).items())
        table.add_row("[red]BUG[/red]", o.concern.value, f"{o.unit_name}:{o.start_line}",
                      f"{o.safety} violable · ce: {ce} · {o.execution_detail}")
    for o in report.proposed:
        ce = ", ".join(f"{k}={v}" for k, v in (o.counterexample or {}).items())
        table.add_row("[magenta]PROPOSED[/magenta]", o.concern.value, f"{o.unit_name}:{o.start_line}",
                      f"generalist: {o.safety} may fail · ce: {ce} · verify translation")
    for o in report.clean:
        table.add_row("[green]SAFE[/green]", o.concern.value, f"{o.unit_name}:{o.start_line}",
                      f"proved {o.safety} (UNSAT)")
    for o in report.inconclusive:
        why = o.execution_detail or o.error or "no decisive result"
        table.add_row("[yellow]UNSURE[/yellow]", o.concern.value, f"{o.unit_name}:{o.start_line}", why)
    console.print(table)

    if report.events:
        stream = "  ".join(
            f"[dim]{e.get('marker','')}[/dim]{e.get('stage','')}"
            + (f":{e.get('concern')}" if e.get("concern") else "")
            for e in report.events
        )
        console.print(f"[dim]stream:[/dim] {stream}")

    console.print(Panel(report.github_comment, title="GitHub-style review comment", expand=True))
    if report.weave_trace_url:
        console.print(f"[blue]weave trace:[/blue] {report.weave_trace_url}")
    console.print()


def _diff_from_git(repo: str, base: str, head: str) -> str:
    return subprocess.run(["git", "diff", f"{base}..{head}"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        console.print(__doc__)
        return

    weave_on = _init_weave()

    if args[0] == "--repo":
        repo, base, head = args[1], args[2] if len(args) > 2 else "main", args[3] if len(args) > 3 else "HEAD"
        analysis = parse_diff_mod.run(_diff_from_git(repo, base, head))
        _print(_run(analysis, weave_on), f"{repo} {base}..{head}")
        return

    target = args[0]
    if target.startswith("demo"):
        names = [target.split(":", 1)[1]] if ":" in target else list(EXAMPLES)
        for name in names:
            analysis = parse_diff_mod.run(EXAMPLES[name])
            _print(_run(analysis, weave_on), f"example: {name}")
        return

    # treat as a path to a diff file
    with open(target) as fh:
        analysis = parse_diff_mod.run(fh.read())
    _print(_run(analysis, weave_on), target)


if __name__ == "__main__":
    main()
