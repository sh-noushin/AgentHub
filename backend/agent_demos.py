"""Demos that run the AgentHub agents and show the whole message trace."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agents import (
    final_answer,
    run_math_agent,
    run_order_agent,
    run_support_agent,
)
from data import ORDERS


def describe(message: BaseMessage) -> str:
    """Turn one message into a short line a beginner can read."""
    if isinstance(message, AIMessage) and message.tool_calls:
        requests = [f"{call['name']}({call['args']})" for call in message.tool_calls]
        return f"AI asks for   : {', '.join(requests)}"
    if isinstance(message, ToolMessage):
        return f"Tool returns  : {message.content}"
    return f"{message.type:14}: {message.content}"


def print_trace(question: str, messages: list[BaseMessage], label: str) -> None:
    """Print one agent run: the question, every message, then the answer."""
    print(f"You: {question}")
    for message in messages:
        print(f"  {describe(message)}")
    print(f"AgentHub ({label}): {final_answer(messages)}")


def demo_math_agent(model: BaseChatModel) -> None:
    """Show the tool-calling loop for one question, step by step."""
    question = "Calculate a 20 percent discount for 100 euros."
    print_trace(question, run_math_agent(model, question), "math agent")


def demo_order_agent(model: BaseChatModel) -> None:
    """Show the same loop with order tools, including a real change to the data."""
    question = "What is the status of order 102?"
    print_trace(question, run_order_agent(model, question), "order agent")

    question = "Cancel order 101."
    print_trace(question, run_order_agent(model, question), "order agent")

    # The tool really changed the dictionary in data.py
    print(f"ORDERS[101] is now {ORDERS[101]}")


def demo_support_agent(model: BaseChatModel) -> None:
    """Show an agent that never calls a tool, so its trace has only three messages."""
    question = "Write a friendly message for a customer whose order is delayed."
    print_trace(question, run_support_agent(model, question), "support agent")
