"""The first AgentHub graph: START -> one node -> END."""

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from prompts import question_prompt


class ChatState(TypedDict):
    """The data that travels through the graph."""

    question: str
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
