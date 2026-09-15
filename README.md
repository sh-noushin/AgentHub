# AgentHub

A beginner-friendly console AI project built step by step with Python, LangChain, LangGraph and MCP.

## Status

Step 18: `ToolNode` and `tools_condition` — the agent loop as a graph.

## Setup

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your real key in `GEMINI_API_KEY`.

## Run

```powershell
cd backend
python main.py
```

Expected output:

```
AgentHub started.
You: My favourite order is order 105.
AgentHub: Got it.
You: Which order did I mention?
AgentHub: Order 105.
You: Please cancel order 105.
AgentHub: action=cancel order_id=105
You: What is the status of order 102?
AgentHub: action=status order_id=102
You: Do you sell umbrellas?
AgentHub: action=other order_id=None
You: What is the capital of France?
AgentHub (StrOutputParser): Paris
You: Please cancel order 105.
AgentHub (PydanticOutputParser): OrderRequest(action='cancel', order_id=105)
add.invoke -> 13
get_order.invoke -> Order 105: status=Processing, user_id=user-1
calculate_discount.invoke -> 80.0
You: Calculate a 20 percent discount for 100 euros.
Model requested: [{'name': 'calculate_discount', 'args': {'price': 100, 'percent': 20}, 'id': '...', 'type': 'tool_call'}]
Model text: ''
AgentHub executed calculate_discount -> 80.0
MathToolkit tools: ['add', 'multiply', 'calculate_discount']
You: Show order 105.
Model requested: []
Model text: 'I can only help with calculations.'
You: What is 6 multiplied by 7?
Graph state in: {'question': 'What is 6 multiplied by 7?'}
AgentHub (graph): 42
Graph state out: ['answer', 'question']
You: Show order 105.
  understand -> OrderRequest(action='show', order_id=105)
  route      -> handle
  result     -> Order 105: status=Processing, user_id=user-1
  reply      -> Order 105 is still being processed.
You: Please cancel my order.
  understand -> OrderRequest(action='cancel', order_id=None)
  route      -> missing_info
  result     -> No order number was mentioned.
  reply      -> Could you tell me the order number?
You: Do you sell umbrellas?
  understand -> OrderRequest(action='other', order_id=None)
  route      -> not_supported
  result     -> This question is not about an order.
  reply      -> Sorry, I can only help with orders.
You: Calculate a 20 percent discount for 100 euros.
  system        : You are the AgentHub math agent. ...
  human         : Calculate a 20 percent discount for 100 euros.
  AI asks for   : calculate_discount({'price': 100, 'percent': 20})
  Tool returns  : 80.0
  ai            : The final price is 80 euros.
AgentHub (math agent): The final price is 80 euros.
You: What is the status of order 102?
  ...
  AI asks for   : get_order_status({'order_id': 102})
  Tool returns  : Shipped
AgentHub (order agent): Order 102 has been shipped.
You: Cancel order 101.
  ...
  AI asks for   : cancel_order({'order_id': 101})
  Tool returns  : Order 101 was cancelled.
AgentHub (order agent): Order 101 is cancelled.
ORDERS[101] is now {'status': 'Cancelled', 'user_id': 'user-1'}
You: Write a friendly message for a customer whose order is delayed.
  system        : You are the AgentHub support agent. ...
  human         : Write a friendly message for a customer whose order is delayed.
  ai            : Sorry about the wait! Your order is on its way ...
AgentHub (support agent): Sorry about the wait! Your order is on its way ...
You: What is 6 multiplied by 7?
  supervisor -> math_agent (It is a multiplication.)
You: Show order 105.
  supervisor -> order_agent (It asks about one order.)
You: Write a friendly delayed-order message.
  supervisor -> support_agent (It asks for a customer message.)
You: Calculate a 20 percent discount for 100 euros.
  supervisor -> math_agent (It is a discount calculation.)
You: What is 6 multiplied by 7?
  supervisor -> math_agent
AgentHub: 6 multiplied by 7 is 42.
You: Show order 105.
  supervisor -> order_agent
AgentHub: Order 105 is still processing.
You: Write a friendly message for a customer whose order is delayed.
  supervisor -> support_agent
AgentHub: Sorry for the wait, your order is on its way!
You: Calculate a 20 percent discount for 100 euros.
  human         : Calculate a 20 percent discount for 100 euros.
  AI asks for   : calculate_discount({'price': 100, 'percent': 20})
  Tool returns  : 80.0
  ai            : The final price is 80 euros.
AgentHub (math graph): The final price is 80 euros.
```
