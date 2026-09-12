"""AgentHub entry point."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from models import build_model
from prompts import SYSTEM_PROMPT, question_prompt


def ask_with_messages(question: str) -> AIMessage:
    """Ask Gemini using a hand-written list of messages."""
    model = build_model()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    return model.invoke(messages)


def ask_with_prompt_template(question: str) -> AIMessage:
    """Ask Gemini using the reusable ChatPromptTemplate."""
    model = build_model()
    prompt_value = question_prompt.invoke({"question": question})
    return model.invoke(prompt_value)


def main() -> None:
    """Start AgentHub and compare both ways of building messages."""
    print("AgentHub started.")

    first_question = "What is 6 multiplied by 7? Answer with the number only."
    first_answer = ask_with_messages(first_question)
    print(f"You: {first_question}")
    print(f"AgentHub: {first_answer.content}")

    second_question = "What is the capital of France?"
    second_answer = ask_with_prompt_template(second_question)
    print(f"You: {second_question}")
    print(f"AgentHub: {second_answer.content}")


# Only run main() when this file is executed directly
if __name__ == "__main__":
    main()
