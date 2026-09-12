# AgentHub

A beginner-friendly console AI project built step by step with Python, LangChain, LangGraph and MCP.

## Status

Step 6: `StrOutputParser` and `PydanticOutputParser`.

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
```
