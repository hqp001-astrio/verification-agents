from __future__ import annotations

import json
import time
from typing import Callable

import openai
from dotenv import load_dotenv

from verification_agents.models import UserSelection, VerifiableProperty, VerificationReport
from verification_agents.prompts import ORCHESTRATOR_SYSTEM_PROMPT, build_initial_message
from verification_agents.tools import ask_user as ask_user_tool
from verification_agents.tools import formalize as formalize_tool
from verification_agents.tools.dispatcher import configure, dispatch
from verification_agents.tools.registry import build_tools

load_dotenv()

ClarificationHandler = Callable[[str, list[VerifiableProperty]], UserSelection]


class OrchestratorAgent:
    def __init__(
        self,
        api_key: str,
        clarification_handler: ClarificationHandler,
        model: str = "gpt-4o",
        max_turns: int = 20,
        z3_timeout_ms: int = 30_000,
    ) -> None:
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_turns = max_turns
        self.z3_timeout_ms = z3_timeout_ms

        ask_user_tool.set_handler(clarification_handler)
        configure(api_key=api_key, model=model)

    def run(self, diff: str, user_intent: str = "") -> VerificationReport:
        formalize_tool.clear_store()
        start = time.monotonic()

        messages: list[dict] = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": build_initial_message(diff, user_intent)},
        ]

        report: VerificationReport | None = None

        for _ in range(self.max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                tools=build_tools(),
                messages=messages,
                max_tokens=4096,
                temperature=0,
            )

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            finish_reason = response.choices[0].finish_reason

            if finish_reason == "stop":
                break

            if finish_reason != "tool_calls" or not msg.tool_calls:
                break

            tool_results = []
            for tc in msg.tool_calls:
                name = tc.function.name
                tool_input = json.loads(tc.function.arguments)

                result_str = dispatch(name, tool_input)

                if name == "submit_report":
                    result_dict = json.loads(result_str)
                    result_dict["elapsed_s"] = time.monotonic() - start
                    report = VerificationReport(**result_dict)

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            messages.extend(tool_results)

            if report is not None:
                return report

        if report is None:
            report = VerificationReport(
                summary="Agent did not complete the verification run.",
                elapsed_s=time.monotonic() - start,
            )

        return report
