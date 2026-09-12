# AgentHub

A beginner-friendly console AI project built step by step with Python, LangChain, LangGraph and MCP.

## Status

Step 2: Gemini connected through LangChain, one question asked.

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
You: What is 6 multiplied by 7? Answer with the number only.
AgentHub: 42
```
