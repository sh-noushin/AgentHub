"""The AgentHub agents. An agent is a model that may call tools in a loop."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from prompts import MATH_SYSTEM_PROMPT
from tools import MathToolkit

# Safety net: stop even if the model keeps asking for more tools
MAX_STEPS = 5


def run_math_agent(model: BaseChatModel, question: str) -> list[BaseMessage]:
    """Answer one math question, running every tool the model asks for."""
    math_tools = MathToolkit().get_tools()
    tools_by_name = {math_tool.name: math_tool for math_tool in math_tools}
    math_model = model.bind_tools(math_tools)

    messages: list[BaseMessage] = [
        SystemMessage(MATH_SYSTEM_PROMPT),
        HumanMessage(question),
    ]

    for _ in range(MAX_STEPS):
        message = math_model.invoke(messages)
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

    messages.append(AIMessage("I could not finish this calculation."))
    return messages


def final_answer(messages: list[BaseMessage]) -> str:
    """Return the text of the last message, which is the agent's answer."""
    return str(messages[-1].content)
