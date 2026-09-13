# AgentHub

A beginner-friendly console AI project built step by step with Python, LangChain, LangGraph and MCP.

## Status

Step 10: the first LangGraph — `State`, `START`, one node, `END`.

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
```
