"""The first AgentHub graph: START -> one node -> END."""

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from models import OrderRequest
from prompts import question_prompt, reply_prompt
from tools import ACTION_TOOLS


class ChatState(TypedDict):
    """The data that travels through the one-node graph."""

    question: str
    answer: str


class OrderState(TypedDict):
    """The data that travels through the order graph, one key per node."""

    question: str
    request: OrderRequest
    result: str
    answer: str


def build_chat_graph(model: BaseChatModel) -> CompiledStateGraph:
    """Build and compile a graph whose single node answers one question."""

    def answer_node(state: ChatState) -> dict:
        """Read the question from the state and return the answer."""
        prompt_value = question_prompt.invoke({"question": state["question"]})
        message = model.invoke(prompt_value)
        # Only the keys we return are written back into the state
        return {"answer": message.content}

    builder = StateGraph(ChatState)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)

    # compile() checks the graph and turns it into something we can invoke
    return builder.compile()


def build_order_graph(model: BaseChatModel) -> CompiledStateGraph:
    """Build a graph that understands, handles and answers one order question."""

    def understand_node(state: OrderState) -> dict:
        """Turn the free-text question into a validated OrderRequest."""
        extractor = model.with_structured_output(OrderRequest)
        return {"request": extractor.invoke(state["question"])}

    def handle_node(state: OrderState) -> dict:
        """Run the order tool that matches the extracted action."""
        request = state["request"]
        order_tool = ACTION_TOOLS.get(request.action)
        if order_tool is None:
            return {"result": "This question is not about an order."}
        if request.order_id is None:
            return {"result": "No order number was mentioned."}
        return {"result": order_tool.invoke({"order_id": request.order_id})}

    def reply_node(state: OrderState) -> dict:
        """Let the model phrase the tool result as a sentence for the customer."""
        prompt_value = reply_prompt.invoke(
            {"question": state["question"], "result": state["result"]}
        )
        return {"answer": model.invoke(prompt_value).content}

    builder = StateGraph(OrderState)
    builder.add_node("understand", understand_node)
    builder.add_node("handle", handle_node)
    builder.add_node("reply", reply_node)

    # One straight path: every question visits all three nodes in this order
    builder.add_edge(START, "understand")
    builder.add_edge("understand", "handle")
    builder.add_edge("handle", "reply")
    builder.add_edge("reply", END)

    return builder.compile()
