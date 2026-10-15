from __future__ import annotations

from energy_agents.agents.base import GROUNDED_SYSTEM, AgentResult, grounded_prompt
from energy_agents.data import EnergyRepository
from energy_agents.knowledge import KnowledgeBase
from energy_agents.llm import ChatModel


class SupportAgent:
    """Handles outages, service questions and energy-saving advice using the knowledge base."""

    name = "support"

    def __init__(self, repo: EnergyRepository, kb: KnowledgeBase, llm: ChatModel) -> None:
        self.repo = repo
        self.kb = kb
        self.llm = llm

    def run(self, question: str, customer_id: str) -> AgentResult:
        customer = self.repo.get_customer(customer_id)
        outages = self.repo.list_outages(customer.area)
        passages = self.kb.search(question, k=3)

        facts = {
            "customer_area": customer.area,
            "rooftop_solar": customer.solar,
            "outages_in_area": outages,
            "knowledge_base": [{"title": p.title, "text": p.text} for p in passages],
        }
        answer = self.llm.complete(GROUNDED_SYSTEM.format(role="the customer-support specialist"),
                                   grounded_prompt(question, facts))
        sources = [f"{p.source}#{p.title}" for p in passages]
        sources += [f"outage:{o['ticket_id']}" for o in outages]
        return AgentResult(self.name, answer.strip(), facts, sources)
