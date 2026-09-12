"""Prompt templates for AgentHub."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

# Same prompt, but earlier turns are inserted between the rules and the question
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
