from __future__ import annotations

import json
from typing import Any

from verification_agents.tools import ask_user, formalize, parse_diff, submit_report, z3_solve

_api_key: str = ""
_model: str = "claude-sonnet-4-6"


def configure(api_key: str, model: str) -> None:
    global _api_key, _model
    _api_key = api_key
    _model = model


def dispatch(name: str, tool_input: dict[str, Any]) -> str:
    """Route a tool call by name and return JSON string result."""
    if name == "parse_diff":
        result = parse_diff.run(tool_input["diff"])
        return result.model_dump_json()

    if name == "ask_user":
        result = ask_user.run(tool_input["question"], tool_input["options"])
        return json.dumps(result)

    if name == "formalize":
        result = formalize.run(
            selected=tool_input["selected"],
            code_analysis=tool_input["code_analysis"],
            api_key=_api_key,
            model=_model,
        )
        return json.dumps(result)

    if name == "z3_solve":
        result = z3_solve.run(tool_input["constraints"])
        return json.dumps(result)

    if name == "submit_report":
        result = submit_report.run(
            results=tool_input["results"],
            summary=tool_input["summary"],
            code_analysis=tool_input["code_analysis"],
            elapsed_s=tool_input.get("elapsed_s", 0.0),
        )
        return json.dumps(result)

    raise ValueError(f"Unknown tool: {name}")
