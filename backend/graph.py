"""The AgentHub graphs: one node, then routed nodes, then a tool-calling loop."""

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from models import OrderRequest
from prompts import MATH_SYSTEM_PROMPT, question_prompt, reply_prompt
from tools import ACTION_TOOLS, MathToolkit


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


def route_action(state: OrderState) -> str:
    """Decide which node runs after understand. Returns a branch name only."""
    request = state["request"]
    if request.action == "other":
        return "not_supported"
    if request.order_id is None:
        return "missing_info"
    return "handle"


def build_order_graph(model: BaseChatModel) -> CompiledStateGraph:
    """Build a graph that understands, routes, handles and answers one question."""

    def understand_node(state: OrderState) -> dict:
        """Turn the free-text question into a validated OrderRequest."""
        extractor = model.with_structured_output(OrderRequest)
        return {"request": extractor.invoke(state["question"])}

    def handle_node(state: OrderState) -> dict:
        """Run the order tool that matches the extracted action."""
        request = state["request"]
        # Routing already guaranteed the action and the order number are usable
        order_tool = ACTION_TOOLS[request.action]
        return {"result": order_tool.invoke({"order_id": request.order_id})}

    def missing_info_node(_state: OrderState) -> dict:
        """Handle questions about an order whose number we never received."""
        return {"result": "No order number was mentioned."}

    def not_supported_node(_state: OrderState) -> dict:
        """Handle questions that are not about an order at all."""
        return {"result": "This question is not about an order."}

    def reply_node(state: OrderState) -> dict:
        """Let the model phrase the tool result as a sentence for the customer."""
        prompt_value = reply_prompt.invoke(
            {"question": state["question"], "result": state["result"]}
        )
        return {"answer": model.invoke(prompt_value).content}

    builder = StateGraph(OrderState)
    builder.add_node("understand", understand_node)
    builder.add_node("handle", handle_node)
    builder.add_node("missing_info", missing_info_node)
    builder.add_node("not_supported", not_supported_node)
    builder.add_node("reply", reply_node)

    builder.add_edge(START, "understand")

    # route_action picks one of these three names, and we go to that node
    builder.add_conditional_edges(
        "understand",
        route_action,
        {
            "handle": "handle",
            "missing_info": "missing_info",
            "not_supported": "not_supported",
        },
    )

    # Whichever branch ran, the answer is always phrased by the same node
    builder.add_edge("handle", "reply")
    builder.add_edge("missing_info", "reply")
    builder.add_edge("not_supported", "reply")
    builder.add_edge("reply", END)

    return builder.compile()


def build_math_agent_graph(model: BaseChatModel) -> CompiledStateGraph:
    """The step-13 math agent rebuilt as a graph, with no loop written by us."""
    math_tools = MathToolkit().get_tools()
    math_model = model.bind_tools(math_tools)

    def agent_node(state: MessagesState) -> dict:
        """Call the model with the rules plus the whole conversation so far."""
        messages = [SystemMessage(MATH_SYSTEM_PROMPT), *state["messages"]]
        # The list we return is appended to the state, not written over it
        return {"messages": [math_model.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    # ToolNode runs every tool the last AIMessage asked for
    builder.add_node("tools", ToolNode(math_tools))

    builder.add_edge(START, "agent")
    # tools_condition returns "tools" when tool calls are present, END otherwise
    builder.add_conditional_edges("agent", tools_condition)
    # The loop: after the tools run, the model reads their results
    builder.add_edge("tools", "agent")

    return builder.compile()
