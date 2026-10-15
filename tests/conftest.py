import json
from pathlib import Path

import pytest

from energy_agents.data import InMemoryRepository
from energy_agents.knowledge import LocalKnowledgeBase
from energy_agents.llm import FakeChat

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo() -> InMemoryRepository:
    return InMemoryRepository.from_json(ROOT / "data" / "customers.json")


@pytest.fixture
def kb() -> LocalKnowledgeBase:
    return LocalKnowledgeBase.from_directory(ROOT / "knowledge_base")


def _route(prompt: str) -> str:
    lowered = prompt.lower()
    agents = []
    if "bill" in lowered:
        agents.append("billing")
    if "usage" in lowered or "using" in lowered:
        agents.append("usage")
    if "outage" in lowered or "power" in lowered:
        agents.append("support")
    return json.dumps({"agents": agents, "reason": "test"})


@pytest.fixture
def fake_llm() -> FakeChat:
    """Routes with simple rules and answers by naming the specialist that was asked."""

    def respond(system: str, prompt: str) -> str:
        if system.startswith("You are the supervisor"):
            return _route(prompt)
        if system.startswith("You combine"):
            return "COMBINED: " + " | ".join(
                line for line in prompt.splitlines() if line.startswith("[")
            )
        role = system.split("You are ")[1].split(" for ")[0]
        return f"answer from {role}"

    return FakeChat(respond)
