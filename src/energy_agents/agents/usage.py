from __future__ import annotations

from energy_agents.agents.base import GROUNDED_SYSTEM, AgentResult, grounded_prompt
from energy_agents.calculations import usage_summary
from energy_agents.data import EnergyRepository
from energy_agents.llm import ChatModel


class UsageAgent:
    """Explains consumption: trends, peaks and month-on-month changes."""

    name = "usage"

    def __init__(self, repo: EnergyRepository, llm: ChatModel) -> None:
        self.repo = repo
        self.llm = llm

    def run(self, question: str, customer_id: str) -> AgentResult:
        customer = self.repo.get_customer(customer_id)
        monthly = self.repo.get_monthly_usage(customer_id)
        facts = {
            "customer": {"name": customer.name, "area": customer.area, "rooftop_solar": customer.solar},
            "usage_summary": usage_summary(monthly),
            "last_6_months_kwh": dict(list(monthly.items())[-6:]),
        }
        answer = self.llm.complete(GROUNDED_SYSTEM.format(role="the energy-usage specialist"),
                                   grounded_prompt(question, facts))
        return AgentResult(self.name, answer.strip(), facts, ["meter_readings"])
