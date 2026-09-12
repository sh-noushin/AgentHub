"""AgentHub entry point."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from models import OrderRequest, build_model
from prompts import chat_prompt


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


def main() -> None:
    """Start AgentHub and run the current demos."""
    print("AgentHub started.")

    model = build_model()
    demo_history(model)
    demo_structured_output(model)


# Only run main() when this file is executed directly
if __name__ == "__main__":
    main()
