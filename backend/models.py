"""Gemini model configuration and data shapes for AgentHub."""

import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

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


class Route(BaseModel):
    """Which agent the supervisor picked, and why."""

    agent: Literal["math_agent", "order_agent", "support_agent"] = Field(
        description=(
            "math_agent for any calculation, order_agent for looking up or "
            "cancelling one order, support_agent for friendly messages"
        )
    )
    reason: str = Field(description="One short sentence explaining the choice")


class OrderRequest(BaseModel):
    """What the customer wants to do with an order."""

    action: Literal["show", "status", "cancel", "other"] = Field(
        description="What the customer wants to do with the order"
    )
    order_id: Optional[int] = Field(
        default=None,
        description="The order number mentioned by the customer, if any",
    )
