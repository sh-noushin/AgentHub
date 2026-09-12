"""Gemini model configuration for AgentHub."""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the .env file into environment variables once, at import time
load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"


def get_api_key() -> str:
    """Read GEMINI_API_KEY from the environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to the .env file in the project root."
        )
    return api_key


def build_model() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model that AgentHub talks to."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=get_api_key(),
        temperature=0,
    )
