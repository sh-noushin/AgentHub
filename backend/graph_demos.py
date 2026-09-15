"""Demos that run the AgentHub graphs and show what the state looks like."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver

from agent_demos import describe
from graph import (
    build_chat_graph,
    build_math_agent_graph,
    build_order_agent_graph,
    build_order_graph,
    route_action,
)


def demo_graph(model: BaseChatModel) -> None:
    """Run the one-node graph and show the state before and after."""
    graph = build_chat_graph(model)

    question = "What is 6 multiplied by 7?"
    start_state = {"question": question}
    final_state = graph.invoke(start_state)

    print(f"You: {question}")
    print(f"Graph state in: {start_state}")
    print(f"AgentHub (graph): {final_state['answer']}")
    print(f"Graph state out: {sorted(final_state)}")


def demo_order_graph(model: BaseChatModel) -> None:
    """Run the routed graph and show which branch each question took."""
    graph = build_order_graph(model)
    questions = [
        "Show order 105.",
        "Please cancel my order.",
        "Do you sell umbrellas?",
    ]

    for question in questions:
        final_state = graph.invoke({"question": question})
        print(f"You: {question}")
        print(f"  understand -> {final_state['request']!r}")
        # The final state still holds the request, so we can ask for the branch again
        print(f"  route      -> {route_action(final_state)}")
        print(f"  result     -> {final_state['result']}")
        print(f"  reply      -> {final_state['answer']}")


def demo_math_agent_graph(model: BaseChatModel) -> None:
    """Run the math agent as a graph, where ToolNode does what our loop did."""
    graph = build_math_agent_graph(model)

    question = "Calculate a 20 percent discount for 100 euros."
    final_state = graph.invoke({"messages": [HumanMessage(question)]})

    print(f"You: {question}")
    for message in final_state["messages"]:
        print(f"  {describe(message)}")
    print(f"AgentHub (math graph): {final_state['messages'][-1].content}")


def demo_memory(model: BaseChatModel) -> None:
    """Two turns in one thread, so the second question can say "it"."""
    graph = build_order_agent_graph(model, InMemorySaver())
    config = {"configurable": {"thread_id": "customer-1"}}

    for question in ["Show order 105.", "Cancel it."]:
        state = graph.invoke({"messages": [HumanMessage(question)]}, config)
        print(f"You (customer-1): {question}")
        print(f"AgentHub: {state['messages'][-1].content}")
        print(f"  messages kept in this thread: {len(state['messages'])}")

    # A different thread_id is a different conversation with an empty history
    other_config = {"configurable": {"thread_id": "customer-2"}}
    state = graph.invoke({"messages": [HumanMessage("Cancel it.")]}, other_config)
    print("You (customer-2): Cancel it.")
    print(f"AgentHub: {state['messages'][-1].content}")
    print(f"  messages kept in this thread: {len(state['messages'])}")
