"""AgentHub entry point."""

from models import build_model


def ask_gemini(question: str) -> str:
    """Send one question to Gemini and return the answer text."""
    model = build_model()
    response = model.invoke(question)
    return response.content


def main() -> None:
    """Start AgentHub and ask Gemini one question."""
    print("AgentHub started.")

    question = "What is 6 multiplied by 7? Answer with the number only."
    answer = ask_gemini(question)

    print(f"You: {question}")
    print(f"AgentHub: {answer}")


# Only run main() when this file is executed directly
if __name__ == "__main__":
    main()
