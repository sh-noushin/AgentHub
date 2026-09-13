"""Prompt templates and output parsers for AgentHub."""

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from models import OrderRequest

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

MATH_SYSTEM_PROMPT = (
    "You are the AgentHub math agent. Use the tools for every calculation, "
    "never do the arithmetic yourself. Answer in one short sentence."
)

ORDER_SYSTEM_PROMPT = (
    "You are the AgentHub order agent. Use the tools to look up or cancel orders. "
    "Never invent order details. If no order number is given, ask for it."
)

SUPPORT_SYSTEM_PROMPT = (
    "You are the AgentHub support agent. Write short, warm messages to customers. "
    "You have no tools and no access to order data, so never state a status or a date. "
    "Apologise once, explain simply, and say what happens next."
)

REPLY_SYSTEM_PROMPT = (
    "You are AgentHub. Reply to the customer in one short, friendly sentence. "
    "Use only the tool result you are given. Never invent order details."
)

# Used by the last node of the order graph to phrase the final answer
reply_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REPLY_SYSTEM_PROMPT),
        ("human", "Question: {question}\nTool result: {result}"),
    ]
)

# Turns an AIMessage into plain text
str_parser = StrOutputParser()

# Turns the model's JSON text into a validated OrderRequest object
order_parser = PydanticOutputParser(pydantic_object=OrderRequest)

EXTRACTION_SYSTEM_PROMPT = (
    "Extract what the customer wants to do with an order.\n"
    "{format_instructions}"
)

# The parser explains its own output format, so it is filled in once, here
extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", "{sentence}"),
    ]
).partial(format_instructions=order_parser.get_format_instructions())
