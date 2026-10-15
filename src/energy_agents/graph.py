"""LangGraph wiring: supervisor → specialists (in parallel) → synthesizer.

    START → supervisor ─┬─ usage   ─┐
                        ├─ billing ─┼─→ synthesize → END
                        └─ support ─┘

The supervisor's plan becomes the conditional edge, so a compound question
("why is my bill high, and is there an outage?") fans out to several
specialists in one super-step and their results merge through a reducer.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from energy_agents.agents import AgentResult, Supervisor, Synthesizer
from energy_agents.data import CustomerNotFoundError


class Specialist(Protocol):
    name: str

    def run(self, question: str, customer_id: str) -> AgentResult: ...


def _merge(left: dict, right: dict) -> dict:
    return {**left, **right}


class ConversationState(TypedDict, total=False):
    question: str
    customer_id: str
    plan: list[str]
    results: Annotated[dict[str, AgentResult], _merge]
    trace: Annotated[list[dict], operator.add]
    answer: str


def build_graph(supervisor: Supervisor, specialists: list[Specialist], synthesizer: Synthesizer) -> Any:
    by_name = {s.name: s for s in specialists}
    graph = StateGraph(ConversationState)

    def supervisor_node(state: ConversationState) -> dict:
        started = time.perf_counter()
        plan = [p for p in supervisor.plan(state["question"]) if p in by_name]
        return {"plan": plan, "trace": [{"node": "supervisor", "plan": plan,
                                         "ms": round((time.perf_counter() - started) * 1000)}]}

    def make_specialist_node(specialist: Specialist):
        def node(state: ConversationState) -> dict:
            started = time.perf_counter()
            try:
                result = specialist.run(state["question"], state["customer_id"])
            except CustomerNotFoundError:
                raise
            except Exception as exc:  # one failing specialist must not sink the whole reply
                result = AgentResult(specialist.name, f"The {specialist.name} service is unavailable "
                                     "right now.", {"error": type(exc).__name__})
            return {"results": {specialist.name: result},
                    "trace": [{"node": specialist.name, "sources": result.sources,
                               "ms": round((time.perf_counter() - started) * 1000)}]}
        return node

    def synthesize_node(state: ConversationState) -> dict:
        results = state.get("results", {})
        ordered = [results[name] for name in state.get("plan", []) if name in results]
        return {"answer": synthesizer.combine(state["question"], ordered),
                "trace": [{"node": "synthesize", "combined": len(ordered)}]}

    def route(state: ConversationState) -> list[str]:
        return state.get("plan") or ["synthesize"]

    graph.add_node("supervisor", supervisor_node)
    for name, specialist in by_name.items():
        graph.add_node(name, make_specialist_node(specialist))
        graph.add_edge(name, "synthesize")
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route, [*by_name, "synthesize"])
    graph.add_edge("synthesize", END)
    return graph.compile()


def ask(app: Any, question: str, customer_id: str) -> ConversationState:
    return app.invoke({"question": question, "customer_id": customer_id, "results": {}, "trace": []})
