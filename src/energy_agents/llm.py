"""Chat model clients: Amazon Bedrock (Converse API), Ollama for local runs, and a fake for tests."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol


class LLMUnavailableError(RuntimeError):
    """The model endpoint could not be reached."""


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


class BedrockChat:
    def __init__(self, model_id: str, region: str | None = None, client: Any | None = None,
                 max_tokens: int = 1024, temperature: float = 0.2) -> None:
        if client is None:
            import boto3

            client = boto3.client("bedrock-runtime", region_name=region)
        self.name = f"bedrock:{model_id}"
        self._model_id = model_id
        self._client = client
        self._config = {"maxTokens": max_tokens, "temperature": temperature}

    def complete(self, system: str, prompt: str) -> str:
        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=self._config,
        )
        return "".join(b.get("text", "") for b in response["output"]["message"]["content"])


class OllamaChat:
    """Local fallback so the whole system runs on a laptop without AWS."""

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434",
                 timeout: float = 120.0) -> None:
        self.name = f"ollama:{model}"
        self._model = model
        self._url = f"{host.rstrip('/')}/api/chat"
        self._timeout = timeout

    def complete(self, system: str, prompt: str) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(self._url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                return json.loads(response.read())["message"]["content"]
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise LLMUnavailableError(
                f"Ollama is not reachable at {self._url}. Start it with 'ollama serve'."
            ) from exc


class FakeChat:
    """Deterministic stand-in for tests: a function decides the reply from the prompts."""

    def __init__(self, responder: Callable[[str, str], str]) -> None:
        self.name = "fake"
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self._responder(system, prompt)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(text: str) -> dict:
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object in model output")
    obj, _ = json.JSONDecoder().raw_decode(candidate[start:])
    if not isinstance(obj, dict):
        raise ValueError("Model output JSON is not an object")
    return obj
