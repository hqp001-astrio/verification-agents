from __future__ import annotations


def _fn(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOLS: list[dict] = [
    _fn("parse_diff", (
        "Parse a unified diff to extract changed functions, build a call graph, "
        "and discover verifiable properties (array bounds, null dereferences, "
        "integer overflow, loop termination, pre/postconditions)."
    ), {
        "type": "object",
        "properties": {
            "diff": {"type": "string", "description": "Raw unified diff text."},
        },
        "required": ["diff"],
    }),
    _fn("ask_user", (
        "Present the discovered verifiable properties to the user and ask which ones "
        "they want checked. Always call this before `formalize`."
    ), {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Question to ask the user."},
            "options": {
                "type": "array",
                "description": "List of verifiable properties the user can choose from.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "kind": {"type": "string"},
                        "unit_name": {"type": "string"},
                        "filename": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "description": {"type": "string"},
                    },
                    "required": ["id", "kind", "unit_name", "filename", "start_line", "description"],
                },
            },
        },
        "required": ["question", "options"],
    }),
    _fn("formalize", (
        "Translate the user-selected properties into Z3 SMT constraints. "
        "Returns a list of Z3 constraints ready to be passed to `z3_solve`."
    ), {
        "type": "object",
        "properties": {
            "selected": {
                "type": "object",
                "description": "UserSelection with selected_ids and optional extra_notes.",
                "properties": {
                    "selected_ids": {"type": "array", "items": {"type": "string"}},
                    "extra_notes": {"type": "string"},
                },
                "required": ["selected_ids"],
            },
            "code_analysis": {
                "type": "object",
                "description": "The CodeAnalysis object returned by parse_diff.",
            },
        },
        "required": ["selected", "code_analysis"],
    }),
    _fn("z3_solve", (
        "Run the Z3 SMT solver on the formalized constraints. "
        "For each constraint P, asserts Not(P) — SAT means P can be violated (bug found), "
        "UNSAT means P always holds."
    ), {
        "type": "object",
        "properties": {
            "constraints": {
                "type": "array",
                "description": "List of Z3Constraint objects from `formalize`.",
                "items": {"type": "object"},
            },
        },
        "required": ["constraints"],
    }),
    _fn("submit_report", (
        "Finalize the verification run. Maps solver results to bugs with severity levels "
        "and assembles the VerificationReport. Calling this ends the session."
    ), {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "description": "List of SolverResult objects from `z3_solve`.",
                "items": {"type": "object"},
            },
            "summary": {
                "type": "string",
                "description": (
                    "Prose summary of findings. Include what was verified, which bugs "
                    "were found, their severity, and the concrete counterexample values."
                ),
            },
            "code_analysis": {
                "type": "object",
                "description": "The CodeAnalysis object from parse_diff.",
            },
            "elapsed_s": {"type": "number", "description": "Total elapsed time in seconds."},
        },
        "required": ["results", "summary", "code_analysis", "elapsed_s"],
    }),
]


def build_tools() -> list[dict]:
    return TOOLS
