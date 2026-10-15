import json

import pytest

from energy_agents.agents import BillingAgent, Supervisor, SupportAgent, Synthesizer, UsageAgent
from energy_agents.data import CustomerNotFoundError, DynamoDBRepository
from energy_agents.factory import build_app
from energy_agents.graph import ask, build_graph
from energy_agents.handler import lambda_handler
from energy_agents.llm import FakeChat


def test_knowledge_base_finds_relevant_section(kb):
    top = kb.search("why did my bill jump into a higher slab")[0]
    assert top.source == "billing_faq.md"
    assert "slab" in top.text.lower()


def test_billing_agent_grounds_answer_in_computed_facts(repo, kb, fake_llm):
    result = BillingAgent(repo, fake_llm, kb).run("Why is my bill higher?", "C1001")
    assert result.facts["change"]["entered_higher_slab"] is True
    assert result.facts["change"]["total_difference"] == 582.82
    # the computed numbers are what the model is given
    assert "1770.27" in fake_llm.calls[-1][1]
    assert any(s.startswith("billing_faq.md#") for s in result.sources)


def test_support_agent_sees_outage_in_customer_area(repo, kb, fake_llm):
    result = SupportAgent(repo, kb, fake_llm).run("Is there a power outage?", "C1002")
    assert result.facts["outages_in_area"][0]["ticket_id"] == "OUT-5531"
    assert "outage:OUT-5531" in result.sources


def test_supervisor_falls_back_to_keywords_on_bad_output():
    supervisor = Supervisor(FakeChat(lambda s, p: "sorry, I cannot decide"))
    assert supervisor.plan("My bill is too high and there is a power cut") == ["billing", "support"]


def test_supervisor_drops_unknown_agents():
    supervisor = Supervisor(FakeChat(lambda s, p: '{"agents": ["billing", "weather", "billing"]}'))
    assert supervisor.plan("anything") == ["billing"]


def test_graph_single_agent(repo, kb, fake_llm):
    app = build_app(llm=fake_llm, repo=repo, kb=kb)
    state = ask(app, "How is my usage trending?", "C1003")
    assert state["plan"] == ["usage"]
    assert state["answer"] == "answer from the energy-usage specialist"


def test_graph_fans_out_and_synthesizes(repo, kb, fake_llm):
    app = build_app(llm=fake_llm, repo=repo, kb=kb)
    state = ask(app, "Why is my bill high and is there a power outage?", "C1002")
    assert state["plan"] == ["billing", "support"]
    assert set(state["results"]) == {"billing", "support"}
    assert state["answer"] == "COMBINED: [billing] | [support]"
    assert [t["node"] for t in state["trace"]][0] == "supervisor"
    assert state["trace"][-1]["node"] == "synthesize"


def test_graph_off_topic_question_gets_clarification(repo, kb, fake_llm):
    app = build_app(llm=fake_llm, repo=repo, kb=kb)
    state = ask(app, "Tell me a joke", "C1001")
    assert state["plan"] == []
    assert "Could you tell me" in state["answer"]


def test_failing_specialist_degrades_gracefully(fake_llm):
    class Broken:
        name = "usage"

        def run(self, question, customer_id):
            raise TimeoutError("db timeout")

    app = build_graph(Supervisor(fake_llm), [Broken()], Synthesizer(fake_llm))
    state = ask(app, "What is my usage?", "C1001")
    assert "unavailable" in state["answer"]


def test_unknown_customer_raises(repo, kb, fake_llm):
    with pytest.raises(CustomerNotFoundError):
        UsageAgent(repo, fake_llm).run("usage?", "NOPE")


def test_lambda_handler(repo, kb, fake_llm):
    app = build_app(llm=fake_llm, repo=repo, kb=kb)
    ok = lambda_handler({"body": json.dumps({"customer_id": "C1001", "question": "my bill?"})}, None, app)
    assert ok["statusCode"] == 200
    assert json.loads(ok["body"])["agents"] == ["billing"]

    missing = lambda_handler({"body": json.dumps({"customer_id": "X", "question": "my bill?"})}, None, app)
    assert missing["statusCode"] == 404

    bad = lambda_handler({"body": "not json"}, None, app)
    assert bad["statusCode"] == 400


def test_dynamodb_repository_reads_single_table():
    class FakeTable:
        def get_item(self, Key):  # noqa: N803
            if Key == {"PK": "CUSTOMER#C9", "SK": "PROFILE"}:
                return {"Item": {"name": "Test", "area": "Hebbal", "plan": "residential", "solar": False}}
            return {}

        def query(self, **kwargs):
            if "ExclusiveStartKey" not in kwargs:
                return {"Items": [{"SK": "USAGE#2026-09", "kwh": 120}], "LastEvaluatedKey": {"k": 1}}
            return {"Items": [{"SK": "USAGE#2026-08", "kwh": 110}]}

    repo = DynamoDBRepository(FakeTable())
    assert repo.get_customer("C9").area == "Hebbal"
    assert repo.get_monthly_usage("C9") == {"2026-08": 110.0, "2026-09": 120.0}
    with pytest.raises(CustomerNotFoundError):
        repo.get_customer("missing")
