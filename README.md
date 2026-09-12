# AgentHub

A beginner-friendly console AI project built step by step with Python, LangChain, LangGraph and MCP.

## Status

Step 4: `MessagesPlaceholder` — conversation history inserted into the prompt.

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
```
