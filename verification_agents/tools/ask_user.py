from __future__ import annotations

from typing import Callable

from verification_agents.models import UserSelection, VerifiableProperty

ClarificationHandler = Callable[[str, list[VerifiableProperty]], UserSelection]

_handler: ClarificationHandler | None = None


def set_handler(handler: ClarificationHandler) -> None:
    global _handler
    _handler = handler


def run(question: str, options: list[dict]) -> dict:
    if _handler is None:
        raise RuntimeError("No clarification_handler set. Call set_handler() before running.")

    props = [VerifiableProperty(**o) for o in options]
    selection = _handler(question, props)
    return selection.model_dump()
