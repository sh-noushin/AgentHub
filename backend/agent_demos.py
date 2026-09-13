"""Demos that run the AgentHub agents and show the whole message trace."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agents import final_answer, run_math_agent


def describe(message: BaseMessage) -> str:
    """Turn one message into a short line a beginner can read."""
    if isinstance(message, AIMessage) and message.tool_calls:
        requests = [f"{call['name']}({call['args']})" for call in message.tool_calls]
        return f"AI asks for   : {', '.join(requests)}"
    if isinstance(message, ToolMessage):
        return f"Tool returns  : {message.content}"
    return f"{message.type:14}: {message.content}"


def demo_math_agent(model: BaseChatModel) -> None:
    """Show the tool-calling loop for one question, step by step."""
    question = "Calculate a 20 percent discount for 100 euros."
    messages = run_math_agent(model, question)

    print(f"You: {question}")
    for message in messages:
        print(f"  {describe(message)}")
    print(f"AgentHub (math agent): {final_answer(messages)}")
