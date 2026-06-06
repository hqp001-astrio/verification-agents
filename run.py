"""Quick runner: verify a local git diff using the OrchestratorAgent."""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from verification_agents.models import UserSelection, VerifiableProperty
from verification_agents.orchestrator import OrchestratorAgent

load_dotenv()


def ask_me(question: str, options: list[VerifiableProperty]) -> UserSelection:
    print(f"\n{question}\n")
    for i, opt in enumerate(options):
        print(f"  {i + 1}. [{opt.kind}] {opt.unit_name}: {opt.description}")
    raw = input("\nEnter numbers to check (e.g. 1,3) or press Enter for all: ").strip()
    if not raw:
        ids = [o.id for o in options]
    else:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]
        ids = [options[idx].id for idx in indices if 0 <= idx < len(options)]
    notes = input("Any extra properties to verify? (press Enter to skip): ").strip()
    return UserSelection(selected_ids=ids, extra_notes=notes)


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
        print("Error: OPENAI_API_KEY not set. Create a .env file or export the variable.")
        sys.exit(1)

    # Default: diff the sample repo; override with CLI args
    repo = sys.argv[1] if len(sys.argv) > 1 else "../sample-buggy-app"
    base = sys.argv[2] if len(sys.argv) > 2 else "main"
    head = sys.argv[3] if len(sys.argv) > 3 else "feature/add-bulk-ops"

    repo_path = str(Path(repo).resolve())
    print(f"Fetching diff: {repo_path}  ({base}..{head})")
    diff = get_diff(repo_path, base, head)

    if not diff.strip():
        print("No diff found between those branches.")
        sys.exit(1)

    print(f"Diff: {diff.count(chr(10))} lines across "
          f"{diff.count('diff --git')} file(s)\n")

    agent = OrchestratorAgent(
        api_key=api_key,
        clarification_handler=ask_me,
    )

    report = agent.run(diff)

    print("\n" + "=" * 60)
    print(f"VERIFICATION REPORT — {len(report.bugs)} bug(s) found")
    print("=" * 60)

    if report.bugs:
        for bug in report.bugs:
            ce = f"  Counterexample: {bug.counterexample}" if bug.counterexample else ""
            print(f"\n[{bug.severity.upper()}] {bug.filename}:{bug.start_line} — {bug.unit_name}")
            print(f"  {bug.description}{ce}")
    else:
        print("No bugs found.")

    if report.clean:
        print(f"\nVerified clean ({len(report.clean)}):")
        for p in report.clean:
            print(f"  ✓ {p.unit_name}: {p.description[:70]}")

    print(f"\nSummary:\n{report.summary}")
    print(f"\nChecked {report.total_properties_checked} properties in {report.elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
