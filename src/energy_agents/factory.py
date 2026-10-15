"""Builds the full app from environment variables.

LLM_PROVIDER   bedrock (default) | ollama
BEDROCK_MODEL_ID / AWS_REGION     when LLM_PROVIDER=bedrock
OLLAMA_MODEL / OLLAMA_HOST        when LLM_PROVIDER=ollama (default qwen2.5:7b)
DATA_SOURCE    local (default) | dynamodb   (+ DYNAMODB_TABLE)
KB_SOURCE      local (default) | opensearch (+ OPENSEARCH_HOST, OPENSEARCH_INDEX)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from energy_agents.agents import BillingAgent, Supervisor, SupportAgent, Synthesizer, UsageAgent
from energy_agents.data import DynamoDBRepository, EnergyRepository, InMemoryRepository
from energy_agents.graph import build_graph
from energy_agents.knowledge import KnowledgeBase, LocalKnowledgeBase, OpenSearchKnowledgeBase
from energy_agents.llm import BedrockChat, ChatModel, OllamaChat

ROOT = Path(__file__).resolve().parents[2]


def make_llm() -> ChatModel:
    provider = os.environ.get("LLM_PROVIDER", "bedrock").lower()
    if provider == "ollama":
        return OllamaChat(os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
                          os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    model_id = os.environ.get("BEDROCK_MODEL_ID")
    if not model_id:
        raise SystemExit("Set BEDROCK_MODEL_ID, or use LLM_PROVIDER=ollama for a local model.")
    return BedrockChat(model_id, region=os.environ.get("AWS_REGION"))


def make_repository() -> EnergyRepository:
    if os.environ.get("DATA_SOURCE", "local") == "dynamodb":
        return DynamoDBRepository.from_name(os.environ["DYNAMODB_TABLE"], os.environ.get("AWS_REGION"))
    return InMemoryRepository.from_json(os.environ.get("DATA_PATH", ROOT / "data" / "customers.json"))


def make_knowledge_base() -> KnowledgeBase:
    if os.environ.get("KB_SOURCE", "local") == "opensearch":
        from opensearchpy import OpenSearch

        client = OpenSearch(hosts=[os.environ["OPENSEARCH_HOST"]], use_ssl=True)
        return OpenSearchKnowledgeBase(client, os.environ.get("OPENSEARCH_INDEX", "energy-kb"))
    return LocalKnowledgeBase.from_directory(os.environ.get("KB_PATH", ROOT / "knowledge_base"))


def build_app(llm: ChatModel | None = None, repo: EnergyRepository | None = None,
              kb: KnowledgeBase | None = None) -> Any:
    llm = llm or make_llm()
    repo = repo or make_repository()
    kb = kb or make_knowledge_base()
    specialists = [UsageAgent(repo, llm), BillingAgent(repo, llm, kb), SupportAgent(repo, kb, llm)]
    return build_graph(Supervisor(llm), specialists, Synthesizer(llm))
