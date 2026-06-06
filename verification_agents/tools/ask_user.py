from __future__ import annotations

from typing import Callable

from verification_agents.models import UserSelection, VerifiableProperty

ClarificationHandler = Callable[[str, list[VerifiableProperty]], UserSelection]


def run_with_handler(handler: ClarificationHandler, question: str, options: list[dict]) -> dict:
    props = [VerifiableProperty(**o) for o in options]
    selection = handler(question, props)
    return selection.model_dump()
