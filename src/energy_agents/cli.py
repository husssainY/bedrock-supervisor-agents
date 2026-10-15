"""Ask the agent team a question from the terminal.

    energy-agents --customer C1001 "Why was my bill so high last month?"
    energy-agents --customer C1002            # interactive mode
"""

from __future__ import annotations

import argparse
import json

from energy_agents.factory import build_app
from energy_agents.graph import ask
from energy_agents.llm import LLMUnavailableError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Supervisor multi-agent assistant for an energy utility")
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--customer", default="C1001", help="Customer ID (sample data: C1001-C1003)")
    parser.add_argument("--trace", action="store_true", help="Print the routing trace")
    args = parser.parse_args(argv)

    app = build_app()

    def answer(question: str) -> None:
        try:
            state = ask(app, question, args.customer)
        except LLMUnavailableError as exc:
            raise SystemExit(f"error: {exc}") from None
        print(f"\n{state['answer']}\n")
        if args.trace:
            print(json.dumps(state["trace"], indent=2))

    if args.question:
        answer(args.question)
        return 0

    print(f"Customer {args.customer}. Type a question, or 'exit' to quit.")
    while (question := input("> ").strip()).lower() not in {"exit", "quit"}:
        if question:
            answer(question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
