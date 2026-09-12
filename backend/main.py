"""AgentHub entry point."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from models import build_model
from prompts import chat_prompt


def ask_with_history(
    model: BaseChatModel,
    question: str,
    history: list[BaseMessage],
) -> AIMessage:
    """Ask Gemini a question together with the earlier turns."""
    prompt_value = chat_prompt.invoke({"history": history, "question": question})
    return model.invoke(prompt_value)


def main() -> None:
    """Start AgentHub and run a short two-turn conversation."""
    print("AgentHub started.")

    model = build_model()
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


# Only run main() when this file is executed directly
if __name__ == "__main__":
    main()
