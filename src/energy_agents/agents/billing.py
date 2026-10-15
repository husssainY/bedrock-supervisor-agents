from __future__ import annotations

from energy_agents.agents.base import GROUNDED_SYSTEM, AgentResult, grounded_prompt
from energy_agents.calculations import compute_bill
from energy_agents.data import EnergyRepository
from energy_agents.knowledge import KnowledgeBase
from energy_agents.llm import ChatModel


class BillingAgent:
    """Explains bills: slab-by-slab breakdown and what changed since last month."""

    name = "billing"

    def __init__(self, repo: EnergyRepository, llm: ChatModel, kb: KnowledgeBase | None = None) -> None:
        self.repo = repo
        self.llm = llm
        self.kb = kb

    def run(self, question: str, customer_id: str) -> AgentResult:
        tariff = self.repo.get_tariff()
        monthly = self.repo.get_monthly_usage(customer_id)
        months = list(monthly)[-2:]
        bills = [compute_bill(m, monthly[m], tariff) for m in months]

        facts: dict = {
            "tariff": {"name": tariff["name"], "currency": tariff["currency"], "slabs": tariff["slabs"],
                       "fixed_charge": tariff["fixed_charge"], "tax_rate": tariff["tax_rate"]},
            "bills": [b.to_dict() for b in bills],
        }
        if len(bills) == 2:
            prev, latest = bills
            facts["change"] = {
                "total_difference": round(latest.total - prev.total, 2),
                "kwh_difference": round(latest.kwh - prev.kwh, 1),
                "entered_higher_slab": latest.top_slab_rate > prev.top_slab_rate,
            }

        sources = ["tariff", "meter_readings"]
        if self.kb is not None:
            passages = self.kb.search(question, k=2)
            facts["policy"] = [{"title": p.title, "text": p.text} for p in passages]
            sources += [f"{p.source}#{p.title}" for p in passages]

        answer = self.llm.complete(GROUNDED_SYSTEM.format(role="the billing specialist"),
                                   grounded_prompt(question, facts))
        return AgentResult(self.name, answer.strip(), facts, sources)
