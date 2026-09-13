"""Demos that run the AgentHub graphs and show what the state looks like."""

from langchain_core.language_models import BaseChatModel

from graph import build_chat_graph, build_order_graph, route_action


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
