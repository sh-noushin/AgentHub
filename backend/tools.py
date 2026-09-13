"""Tools the model can ask AgentHub to run."""

from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool, tool
from pydantic import BaseModel, Field

from data import ORDERS


@tool
def add(a: int, b: int) -> int:
    """Add two numbers and return the sum."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the product."""
    return a * b


@tool
def get_order(order_id: int) -> str:
    """Look up one order and return its status and owner."""
    order = ORDERS.get(order_id)
    if order is None:
        return f"Order {order_id} was not found."
    return f"Order {order_id}: status={order['status']}, user_id={order['user_id']}"


@tool
def get_order_status(order_id: int) -> str:
    """Return only the status of one order."""
    order = ORDERS.get(order_id)
    if order is None:
        return f"Order {order_id} was not found."
    return order["status"]


@tool
def cancel_order(order_id: int) -> str:
    """Cancel one order and return what happened."""
    order = ORDERS.get(order_id)
    if order is None:
        return f"Order {order_id} was not found."
    if order["status"] == "Shipped":
        return f"Order {order_id} was already shipped and cannot be cancelled."
    order["status"] = "Cancelled"
    return f"Order {order_id} was cancelled."


class DiscountInput(BaseModel):
    """Arguments the model must provide to calculate a discount."""

    price: float = Field(description="The original price in euros")
    percent: float = Field(
        ge=0,
        le=100,
        description="The discount percentage, for example 20 for 20 percent",
    )


def calculate_discount(price: float, percent: float) -> float:
    """Return the price after subtracting the discount percentage."""
    discount = price * percent / 100
    return round(price - discount, 2)


# Same result as @tool, but the pieces are passed explicitly
calculate_discount_tool = StructuredTool.from_function(
    func=calculate_discount,
    name="calculate_discount",
    description="Calculate the final price after applying a percentage discount.",
    args_schema=DiscountInput,
)


class MathToolkit(BaseToolkit):
    """Groups every tool that does arithmetic."""

    def get_tools(self) -> list[BaseTool]:
        """Return the tools the math agent is allowed to use."""
        return [add, multiply, calculate_discount_tool]


ORDER_TOOLS = [get_order, get_order_status, cancel_order]

# Maps an OrderRequest action to the tool that performs it
ACTION_TOOLS: dict[str, BaseTool] = {
    "show": get_order,
    "status": get_order_status,
    "cancel": cancel_order,
}

ALL_TOOLS = MathToolkit().get_tools() + ORDER_TOOLS

# Lets us find the right tool when the model asks for one by name
TOOLS_BY_NAME = {single_tool.name: single_tool for single_tool in ALL_TOOLS}
