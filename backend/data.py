"""Fake order data for AgentHub."""

# order_id -> order details. Replaced by a real database only if we ever need one.
ORDERS: dict[int, dict[str, str]] = {
    101: {"status": "Processing", "user_id": "user-1"},
    102: {"status": "Shipped", "user_id": "user-1"},
    105: {"status": "Processing", "user_id": "user-1"},
}
