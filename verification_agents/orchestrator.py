from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import weave
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from verification_agents.models import CodeUnit, Language, UserSelection, VerifiableProperty, VerificationReport
from verification_agents.prompts import ORCHESTRATOR_SYSTEM_PROMPT, build_initial_message, build_preanalyzed_message
from verification_agents.specialists.challenger import challenge as _challenger_challenge
from verification_agents.tools import ask_user as _ask_user_mod
from verification_agents.tools import formalize as _formalize_mod
from verification_agents.tools import parse_diff as _parse_diff_mod
from verification_agents.tools import submit_report as _submit_report_mod
from verification_agents.tools import z3_solve as _z3_solve_mod

load_dotenv()

ClarificationHandler = Callable[[str, list[VerifiableProperty]], UserSelection]


def _build_tools(
    api_key: str,
    model: str,
    handler: ClarificationHandler,
    z3_timeout_ms: int,
    start_time: float,
    report_box: list[VerificationReport | None],
) -> list:

    # --- Verification tools ---

    @tool
    def parse_diff(diff: str) -> str:
        """Parse a unified diff to extract changed functions, build a call graph, and discover verifiable properties (array bounds, null dereferences, integer overflow, loop termination)."""
        return _parse_diff_mod.run(diff).model_dump_json()

    @tool
    def ask_user(question: str, options: list[dict[str, Any]]) -> str:
        """Present the discovered verifiable properties to the user and ask which ones to check. Always call this before formalize."""
        result = _ask_user_mod.run_with_handler(handler, question, options)
        return json.dumps(result)

    @tool
    def formalize(selected: dict[str, Any], code_analysis: dict[str, Any]) -> str:
        """Translate user-selected properties into Z3 SMT constraints. selected is a UserSelection dict with selected_ids; code_analysis is the CodeAnalysis dict from parse_diff."""
        result = _formalize_mod.run(
            selected=selected,
            code_analysis=code_analysis,
            api_key=api_key,
            model=model,
        )
        return json.dumps(result)

    @tool
    def z3_solve(constraints: list[dict[str, Any]]) -> str:
        """Run the Z3 SMT solver on formalized constraints. SAT means the property can be violated (bug found); UNSAT means it always holds."""
        result = _z3_solve_mod.run(constraints, timeout_ms=z3_timeout_ms)
        return json.dumps(result)

    @tool
    def submit_report(
        results: list[dict[str, Any]],
        summary: str,
        code_analysis: dict[str, Any],
        elapsed_s: float,
    ) -> str:
        """Finalize the verification run. Maps solver results to bugs with severity and assembles the VerificationReport. Call this to end the session."""
        result_dict = _submit_report_mod.run(
            results=results,
            summary=summary,
            code_analysis=code_analysis,
            elapsed_s=time.monotonic() - start_time,
        )
        report_box[0] = VerificationReport(**result_dict)
        return json.dumps(result_dict)

    # --- Code tools (Claude Code-style: shell, file read/write/list) ---

    @tool
    def shell(command: str) -> str:
        """Run a bash command and return stdout + stderr. Use to run tests, check git state, execute static analysis, or gather context beyond the diff."""
        proc = subprocess.run(
            command,
            shell=True,  # noqa: S602
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = proc.stdout + proc.stderr
        return output[:8000] if len(output) > 8000 else output

    @tool
    def read_file(path: str) -> str:
        """Read the full contents of a source file. Use to get context beyond what the diff shows."""
        try:
            return Path(path).read_text()
        except Exception as exc:
            return f"Error: {exc}"

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file (e.g. save a patch or a report)."""
        try:
            Path(path).write_text(content)
            return f"Written {len(content)} bytes to {path}"
        except Exception as exc:
            return f"Error: {exc}"

    @tool
    def list_directory(path: str = ".") -> str:
        """List files and subdirectories at the given path."""
        try:
            entries = sorted(Path(path).iterdir(), key=lambda p: (p.is_file(), p.name))
            return "\n".join(f"{'F' if e.is_file() else 'D'}  {e.name}" for e in entries)
        except Exception as exc:
            return f"Error: {exc}"

    @tool
    def challenge_finding(safety: str, reachable: str, unit_source: str = "") -> str:
        """Call the CHALLENGER agent to review a SAT finding before reporting it as a bug.

        The Challenger is an independent LLM that decides whether the safety predicate
        faithfully formalizes a real concern in the code, or whether it is a mistranslation
        (wrong variables, inverted logic, trivially guaranteed by structure).

        Call this for EVERY SAT result from z3_solve before calling submit_report.
        Pass the safety predicate, the reachable-state assumption, and the source of the
        function under review. Returns {valid: bool, issue: str}.
        """
        unit = CodeUnit(
            name="reviewed_function",
            filename="unknown",
            language=Language.UNKNOWN,
            start_line=0,
            end_line=0,
            source=unit_source,
        ) if unit_source.strip() else None
        result = _challenger_challenge(safety, reachable, unit, api_key)
        return json.dumps({"valid": result.valid, "issue": result.issue,
                           "verdict": "CONFIRMED BUG" if result.valid else f"REJECTED: {result.issue}"})

    return [parse_diff, ask_user, formalize, z3_solve, challenge_finding, submit_report,
            shell, read_file, write_file, list_directory]


class OrchestratorAgent:
    def __init__(
        self,
        api_key: str,
        clarification_handler: ClarificationHandler,
        model: str = "gpt-4o-mini",
        max_turns: int = 50,
        z3_timeout_ms: int = 30_000,
        weave_project: str | None = "astrio/verification-agents",
    ) -> None:
        self.api_key = api_key
        self.clarification_handler = clarification_handler
        self.model = model
        self.max_turns = max_turns
        self.z3_timeout_ms = z3_timeout_ms
        self._weave_enabled = False
        if weave_project:
            try:
                weave.init(weave_project)
                self._weave_enabled = True
            except Exception as exc:
                print(f"[weave] tracing disabled: {exc}")

    @weave.op()
    def run(
        self,
        diff: str,
        user_intent: str = "",
        code_analysis: Any | None = None,
        user_selection: Any | None = None,
    ) -> VerificationReport:
        _formalize_mod.clear_store()
        start = time.monotonic()
        report_box: list[VerificationReport | None] = [None]

        tools = _build_tools(
            api_key=self.api_key,
            model=self.model,
            handler=self.clarification_handler,
            z3_timeout_ms=self.z3_timeout_ms,
            start_time=start,
            report_box=report_box,
        )

        llm = ChatOpenAI(api_key=self.api_key, model=self.model, temperature=0)

        graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        )

        if code_analysis is not None and user_selection is not None:
            import json as _json  # noqa: PLC0415
            initial_content = build_preanalyzed_message(
                _json.dumps(code_analysis),
                _json.dumps(user_selection),
            )
        else:
            initial_content = build_initial_message(diff, user_intent)

        invoke_config: dict[str, Any] = {"recursion_limit": self.max_turns * 6}
        if self._weave_enabled:
            from weave.integrations.langchain import WeaveTracer  # noqa: PLC0415

            invoke_config["callbacks"] = [WeaveTracer()]

        import time as _time  # noqa: PLC0415
        for _attempt in range(4):
            try:
                graph.invoke(
                    {"messages": [{"role": "user", "content": initial_content}]},
                    config=invoke_config,
                )
                break
            except Exception as _exc:
                if ("rate_limit" in str(_exc).lower() or "429" in str(_exc)) and _attempt < 3:
                    _time.sleep(2 ** _attempt * 5)
                    continue
                raise

        if report_box[0] is None:
            return VerificationReport(
                summary="Agent did not complete the verification run.",
                elapsed_s=time.monotonic() - start,
            )

        return report_box[0]
