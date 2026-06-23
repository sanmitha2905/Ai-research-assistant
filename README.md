# 🤖 AI Research Assistant

A production-grade multi-agent AI system that autonomously researches, reviews, and synthesizes information into polished answers.

## Architecture

Three specialized LLM agents orchestrated by a deterministic supervisor:

- **🔍 Researcher** — ReAct agent with web search tool. Gathers facts from live sources.
- **🧐 Critic** — Tool-equipped reviewer. Independently re-verifies research claims via search before approving.
- **✍️ Writer** — Synthesizes approved findings into a clear, polished final answer.

## Key Features

- **Live Progress UI** — Real-time status updates show which agent is working (Researching → Reviewing → Writing)
- **Fail-Safe Design** — Multiple safeguards prevent infinite loops, handle iteration limits, detect failed sub-agents gracefully
- **Deterministic Routing** — Supervisor uses explicit state flags, not LLM calls, for fast, reliable control flow
- **Production Ready** — Deployed on Streamlit Cloud, handles edge cases (researcher failures, critic retries, context drift)

## Tech Stack

- **LangGraph** — Multi-agent orchestration with checkpointing
- **LangChain (Classic)** — ReAct agents with tools
- **Groq LLaMA 3.3** — Fast LLM inference
- **DuckDuckGo Search** — Real-time web search
- **Streamlit** — Interactive UI with live streaming updates

## How It Works

1. User asks a question
2. **Researcher** searches the web (up to 5 iterations), compiles findings
3. **Critic** verifies claims via independent searches (up to 5 iterations), gives verdict
4. If research is rejected, loop back to step 2 (max 2 attempts)
5. **Writer** synthesizes approved research into final answer
6. Return to user with full audit trail

## Running Locally

```bash
git clone https://github.com/sanmitha2905/ai-research-assistant.git
cd ai-research-assistant
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
streamlit run app.py
```

## Learning Path (Days 8-13)

- **Day 8-9:** Single agent with tools (ReAct framework, multi-tool fallback)
- **Day 10-11:** LangGraph foundations, persistent memory with checkpointers
- **Day 12:** Multi-agent supervisor pattern, `Command` object routing
- **Day 13:** 3-agent system (Researcher + Critic + Writer), live streaming UI

## Future Improvements

- Add fact-checker agent to verify final answer claims
- Implement human-in-the-loop approval steps
- Support custom knowledge bases (RAG + web search hybrid)
- Deploy to production with rate limiting and cost controls

