"""AgentHub entry point."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent_demos import demo_math_agent
from graph_demos import demo_graph, demo_order_graph
from models import OrderRequest, build_model
from prompts import (
    chat_prompt,
    extraction_prompt,
    order_parser,
    question_prompt,
    str_parser,
)
from tools import (
    ALL_TOOLS,
    TOOLS_BY_NAME,
    MathToolkit,
    add,
    calculate_discount_tool,
    get_order,
)


def ask_with_history(
    model: BaseChatModel,
    question: str,
    history: list[BaseMessage],
) -> AIMessage:
    """Ask Gemini a question together with the earlier turns."""
    prompt_value = chat_prompt.invoke({"history": history, "question": question})
    return model.invoke(prompt_value)


def extract_order_request(model: BaseChatModel, sentence: str) -> OrderRequest:
    """Turn a customer sentence into a validated OrderRequest object."""
    extractor = model.with_structured_output(OrderRequest)
    return extractor.invoke(sentence)


def answer_as_text(model: BaseChatModel, question: str) -> str:
    """Ask a question and get plain text back instead of an AIMessage."""
    prompt_value = question_prompt.invoke({"question": question})
    message = model.invoke(prompt_value)
    return str_parser.invoke(message)


def extract_with_parser(model: BaseChatModel, sentence: str) -> OrderRequest:
    """Extract an OrderRequest by asking for JSON and parsing the text."""
    prompt_value = extraction_prompt.invoke({"sentence": sentence})
    message = model.invoke(prompt_value)
    return order_parser.invoke(message)


def demo_history(model: BaseChatModel) -> None:
    """Run a short two-turn conversation to show history in action."""
    history: list[BaseMessage] = []
    questions = [
        "My favourite order is order 105.",
        "Which order did I mention?",
    ]

    for question in questions:
        answer = ask_with_history(model, question, history)
        print(f"You: {question}")
        print(f"AgentHub: {answer.content}")

        # Keep both turns so the next question can refer back to them
        history.append(HumanMessage(content=question))
        history.append(answer)


def demo_structured_output(model: BaseChatModel) -> None:
    """Show the model returning a Pydantic object instead of free text."""
    sentences = [
        "Please cancel order 105.",
        "What is the status of order 102?",
        "Do you sell umbrellas?",
    ]

    for sentence in sentences:
        request = extract_order_request(model, sentence)
        print(f"You: {sentence}")
        print(f"AgentHub: action={request.action} order_id={request.order_id}")


def demo_parsers(model: BaseChatModel) -> None:
    """Show StrOutputParser and PydanticOutputParser side by side."""
    question = "What is the capital of France?"
    print(f"You: {question}")
    print(f"AgentHub (StrOutputParser): {answer_as_text(model, question)}")

    sentence = "Please cancel order 105."
    request = extract_with_parser(model, sentence)
    print(f"You: {sentence}")
    print(f"AgentHub (PydanticOutputParser): {request!r}")


def demo_tools(model: BaseChatModel) -> None:
    """Show a tool run directly, then a tool requested by the model."""
    # 1. We call the tools ourselves. No model is involved.
    print(f"add.invoke -> {add.invoke({'a': 6, 'b': 7})}")
    print(f"get_order.invoke -> {get_order.invoke({'order_id': 105})}")

    # 2. The StructuredTool is used exactly like the decorated tools.
    price = calculate_discount_tool.invoke({"price": 100, "percent": 20})
    print(f"calculate_discount.invoke -> {price}")

    # 3. The model only asks for a tool. It cannot run Python.
    model_with_tools = model.bind_tools(ALL_TOOLS)
    question = "Calculate a 20 percent discount for 100 euros."
    message = model_with_tools.invoke(question)

    print(f"You: {question}")
    print(f"Model requested: {message.tool_calls}")
    print(f"Model text: {message.content!r}")

    # 4. We look up the requested tool and run it.
    for tool_call in message.tool_calls:
        requested_tool = TOOLS_BY_NAME[tool_call["name"]]
        result = requested_tool.invoke(tool_call["args"])
        print(f"AgentHub executed {tool_call['name']} -> {result}")


def demo_toolkit(model: BaseChatModel) -> None:
    """Bind only the math tools, so the model cannot touch orders."""
    math_tools = MathToolkit().get_tools()
    print(f"MathToolkit tools: {[t.name for t in math_tools]}")

    math_model = model.bind_tools(math_tools)
    question = "Show order 105."
    message = math_model.invoke(question)

    print(f"You: {question}")
    print(f"Model requested: {message.tool_calls}")
    print(f"Model text: {message.content!r}")


def main() -> None:
    """Start AgentHub and run the current demos."""
    print("AgentHub started.")

    model = build_model()
    demo_history(model)
    demo_structured_output(model)
    demo_parsers(model)
    demo_tools(model)
    demo_toolkit(model)
    demo_graph(model)
    demo_order_graph(model)
    demo_math_agent(model)


# Only run main() when this file is executed directly
if __name__ == "__main__":
    main()
