from __future__ import annotations

import json
from dataclasses import dataclass, field

GROUNDED_SYSTEM = (
    "You are {role} for an electricity utility. Answer the customer's question using ONLY the "
    "facts provided. Quote numbers exactly as given. If the facts do not answer the question, "
    "say what is missing. Be concise: at most 5 sentences, plain language, no markdown headings."
)


@dataclass
class AgentResult:
    agent: str
    answer: str
    facts: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


def grounded_prompt(question: str, facts: dict) -> str:
    return f"Facts (JSON):\n{json.dumps(facts, indent=2, default=str)}\n\nCustomer question: {question}"
