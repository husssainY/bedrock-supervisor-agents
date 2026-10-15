"""The supervisor plans which specialists to call; the synthesizer merges their answers."""

from __future__ import annotations

import logging

from energy_agents.agents.base import AgentResult
from energy_agents.llm import ChatModel, parse_json_object

log = logging.getLogger(__name__)

AGENT_NAMES = ("usage", "billing", "support")

ROUTER_SYSTEM = """You are the supervisor of a team of utility customer-service specialists.
Pick every specialist needed to fully answer the customer's question:
- usage: electricity consumption, trends, peaks, comparisons between months
- billing: bill amounts, charges, tariffs, slabs, due dates, disputes, payments
- support: power outages, faults, safety, service requests, energy-saving tips, solar
Reply with JSON only: {"agents": ["..."], "reason": "<one short sentence>"}
Use an empty list only if the question has nothing to do with electricity service."""

# Used when the model's plan cannot be parsed, so a customer always gets an answer.
FALLBACK_KEYWORDS = {
    "usage": ("usage", "consumption", "units", "kwh", "using more", "trend", "peak"),
    "billing": ("bill", "charge", "tariff", "slab", "pay", "due", "amount", "cost", "expensive"),
    "support": ("outage", "power cut", "no power", "fault", "restore", "safety", "save", "solar",
                "tips", "reduce"),
}

SYNTH_SYSTEM = (
    "You combine answers from specialist agents into one reply to a utility customer. "
    "Keep every number exactly as given, remove repetition, and answer in at most 6 sentences."
)


class Supervisor:
    def __init__(self, llm: ChatModel) -> None:
        self.llm = llm

    def plan(self, question: str) -> list[str]:
        try:
            parsed = parse_json_object(self.llm.complete(ROUTER_SYSTEM, question))
            agents = [a for a in parsed.get("agents", []) if a in AGENT_NAMES]
            return list(dict.fromkeys(agents))  # dedupe, keep order
        except (ValueError, AttributeError, TypeError):
            log.warning("router output unparseable; using keyword fallback")
            return self.fallback_plan(question)

    @staticmethod
    def fallback_plan(question: str) -> list[str]:
        lowered = question.lower()
        plan = [name for name, words in FALLBACK_KEYWORDS.items() if any(w in lowered for w in words)]
        return plan or ["support"]


class Synthesizer:
    def __init__(self, llm: ChatModel) -> None:
        self.llm = llm

    def combine(self, question: str, results: list[AgentResult]) -> str:
        if not results:
            return ("I can help with your electricity usage, bills, outages and energy-saving tips. "
                    "Could you tell me a bit more about what you need?")
        if len(results) == 1:
            return results[0].answer
        sections = "\n\n".join(f"[{r.agent}]\n{r.answer}" for r in results)
        return self.llm.complete(SYNTH_SYSTEM, f"Customer question: {question}\n\n{sections}").strip()
