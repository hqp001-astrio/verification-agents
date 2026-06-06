"""Quick runner: verify a local git diff using the OrchestratorAgent."""

import os
import subprocess
import sys
from pathlib import Path

import questionary
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from verification_agents.models import UserSelection, VerifiableProperty
from verification_agents.orchestrator import OrchestratorAgent
from verification_agents.tools import parse_diff as _parse_diff_mod

load_dotenv()

console = Console()

SEVERITY_COLORS = {
    "critical": "bold red",
    "high": "red",
    "medium": "yellow",
    "low": "cyan",
}


def make_ask_me(status):
    def ask_me(question: str, options: list[VerifiableProperty]) -> UserSelection:
        status.stop()
        console.print(f"\n[bold]{question}[/bold]\n")

        choices = [
            questionary.Choice(
                title=f"[{opt.kind}] {opt.unit_name}: {opt.description}",
                value=opt.id,
                checked=True,
            )
            for opt in options
        ]

        selected_ids = questionary.checkbox(
            "Select properties to verify (Space to toggle, Enter to confirm):",
            choices=choices,
        ).ask()

        if selected_ids is None:
            selected_ids = [o.id for o in options]

        notes = questionary.text(
            "Any extra properties to verify? (press Enter to skip):",
            default="",
        ).ask() or ""

        status.start()
        return UserSelection(selected_ids=selected_ids, extra_notes=notes)

    return ask_me


def get_diff(repo_path: str, base: str = "main", head: str = "HEAD") -> str:
    result = subprocess.run(
        ["git", "diff", f"{base}..{head}"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not set. Create a .env file or export the variable.")
        sys.exit(1)

    repo = sys.argv[1] if len(sys.argv) > 1 else "../sample-buggy-app"
    base = sys.argv[2] if len(sys.argv) > 2 else "main"
    head = sys.argv[3] if len(sys.argv) > 3 else "feature/add-bulk-ops"

    repo_path = str(Path(repo).resolve())
    console.print(f"[dim]Fetching diff:[/dim] {repo_path}  [dim]({base}..{head})[/dim]")
    diff = get_diff(repo_path, base, head)

    if not diff.strip():
        console.print("[yellow]No diff found between those branches.[/yellow]")
        sys.exit(1)

    n_lines = diff.count("\n")
    n_files = diff.count("diff --git")
    console.print(f"[dim]Diff:[/dim] {n_lines} lines across {n_files} file(s)\n")

    console.print("[dim]Parsing diff…[/dim]")
    code_analysis = _parse_diff_mod.run(diff)
    console.print(f"[dim]Found {len(code_analysis.properties)} verifiable properties across {len(code_analysis.units)} function(s)[/dim]\n")

    status = console.status("[bold green]Running Z3 + LLM verification…[/bold green]", spinner="dots")
    user_selection = make_ask_me(status)(
        "Which properties do you want to verify?",
        code_analysis.properties,
    )

    agent = OrchestratorAgent(
        api_key=api_key,
        clarification_handler=make_ask_me(status),
    )

    status.start()
    report = agent.run(
        diff,
        code_analysis=code_analysis.model_dump(),
        user_selection=user_selection.model_dump(),
    )
    status.stop()

    _print_report(report)


def _print_report(report) -> None:
    n_bugs = len(report.bugs)
    title_color = "red" if n_bugs else "green"
    title = f"[{title_color}]VERIFICATION REPORT — {n_bugs} bug(s) found[/{title_color}]"
    console.print(Panel(Text.from_markup(title), expand=False))

    if report.bugs:
        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("Severity", style="bold", width=10)
        table.add_column("Location", width=30)
        table.add_column("Description")

        for bug in report.bugs:
            sev = bug.severity.value if hasattr(bug.severity, "value") else str(bug.severity)
            color = SEVERITY_COLORS.get(sev.lower(), "white")
            desc = bug.description
            if bug.counterexample:
                desc += f"\n[dim]Counterexample: {bug.counterexample}[/dim]"
            table.add_row(
                f"[{color}]{sev.upper()}[/{color}]",
                f"{bug.filename}:{bug.start_line}\n[dim]{bug.unit_name}[/dim]",
                desc,
            )
        console.print(table)
    else:
        console.print("[green]No bugs found.[/green]")

    if report.clean:
        console.print(f"\n[green]Verified clean ({len(report.clean)}):[/green]")
        for p in report.clean:
            console.print(f"  [green]✓[/green] {p.unit_name}: {p.description[:70]}")

    console.print(f"\n[bold]Summary:[/bold]\n{report.summary}")
    console.print(f"\n[dim]Checked {report.total_properties_checked} properties in {report.elapsed_s:.1f}s[/dim]")


if __name__ == "__main__":
    main()
