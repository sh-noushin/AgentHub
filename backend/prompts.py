"""Prompt templates for AgentHub."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are AgentHub, a helpful assistant for a small online store. "
    "Answer briefly and clearly. If you do not know something, say so."
)

# Reusable prompt: fixed system rules + one question filled in at runtime
question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)
