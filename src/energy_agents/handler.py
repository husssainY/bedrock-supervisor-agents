"""AWS Lambda entry point behind an HTTP API: POST {"customer_id": "...", "question": "..."}."""

from __future__ import annotations

import json
from typing import Any

from energy_agents.data import CustomerNotFoundError
from energy_agents.graph import ask

_app: Any = None


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, default=str)}


def lambda_handler(event: dict, context: Any, app: Any = None) -> dict:
    global _app
    if app is None:
        if _app is None:
            from energy_agents.factory import build_app

            _app = build_app()
        app = _app

    try:
        payload = json.loads(event.get("body") or "{}")
        customer_id, question = payload["customer_id"], payload["question"].strip()
    except (json.JSONDecodeError, KeyError, AttributeError):
        return _response(400, {"error": "Body must be JSON with 'customer_id' and 'question'"})
    if not question:
        return _response(400, {"error": "Question is empty"})

    try:
        state = ask(app, question, customer_id)
    except CustomerNotFoundError:
        return _response(404, {"error": f"Unknown customer {customer_id}"})

    return _response(200, {"answer": state["answer"], "agents": state.get("plan", []),
                           "trace": state.get("trace", [])})
