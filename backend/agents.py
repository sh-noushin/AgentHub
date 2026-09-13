"""The AgentHub agents. An agent is a model that may call tools in a loop."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_core.tools import BaseTool

from prompts import MATH_SYSTEM_PROMPT, ORDER_SYSTEM_PROMPT
from tools import ORDER_TOOLS, MathToolkit

# Safety net: stop even if the model keeps asking for more tools
MAX_STEPS = 5


def run_agent(
    model: BaseChatModel,
    agent_tools: list[BaseTool],
    system_prompt: str,
    question: str,
) -> list[BaseMessage]:
    """Run one tool-calling loop. Only the tools and the prompt differ per agent."""
    tools_by_name = {agent_tool.name: agent_tool for agent_tool in agent_tools}
    model_with_tools = model.bind_tools(agent_tools)

    messages: list[BaseMessage] = [
        SystemMessage(system_prompt),
        HumanMessage(question),
    ]

    for _ in range(MAX_STEPS):
        message = model_with_tools.invoke(messages)
        messages.append(message)

        # No tool request means the model is done and this is the final answer
        if not message.tool_calls:
            return messages

        for tool_call in message.tool_calls:
            requested_tool = tools_by_name[tool_call["name"]]
            result = requested_tool.invoke(tool_call["args"])
            # The id links the result back to the request the model made
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    messages.append(AIMessage("I could not finish this request."))
    return messages


def run_math_agent(model: BaseChatModel, question: str) -> list[BaseMessage]:
    """The math agent: arithmetic only, so it cannot touch orders."""
    return run_agent(model, MathToolkit().get_tools(), MATH_SYSTEM_PROMPT, question)


def run_order_agent(model: BaseChatModel, question: str) -> list[BaseMessage]:
    """The order agent: looks up and cancels orders, but cannot do arithmetic."""
    return run_agent(model, ORDER_TOOLS, ORDER_SYSTEM_PROMPT, question)


def final_answer(messages: list[BaseMessage]) -> str:
    """Return the text of the last message, which is the agent's answer."""
    return str(messages[-1].content)
