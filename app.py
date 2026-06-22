import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, List, Annotated, Literal
from ddgs import DDGS
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

# =============================================
# TOOL
# =============================================
@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    return "\n".join([f"{r['title']}: {r['body']}" for r in results])

# =============================================
# STATE
# =============================================
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    original_question: str
    loop_count: int
    research_done: bool
    research_attempts: int
    research_approved: bool
    critic_feedback: str
    final_answer_given: bool

# =============================================
# SUPERVISOR
# =============================================
def supervisor_node(state: AgentState) -> Command[Literal["researcher", "critic", "writer", "__end__"]]:
    if state.get("final_answer_given"):
        return Command(goto=END)
    if state.get("loop_count", 0) >= 6:
        return Command(goto=END)
    if not state.get("research_done"):
        return Command(goto="researcher", update={"loop_count": state.get("loop_count", 0) + 1})
    if not state.get("research_approved"):
        return Command(goto="critic", update={"loop_count": state.get("loop_count", 0) + 1})
    return Command(goto="writer", update={"loop_count": state.get("loop_count", 0) + 1})

# =============================================
# RESEARCHER
# =============================================
researcher_prompt = PromptTemplate.from_template("""
Answer the question using the web_search tool to find current facts.
If feedback from a previous review is provided, address it specifically.

Tools available: {tools}
Tool names: {tool_names}

Use this format:
Question: the input question
Thought: think about what to search
Action: the tool to use
Action Input: the search query
Observation: the result
... (repeat if needed)
Thought: I have enough information
Final Answer: the facts found (just the facts, not a polished essay)

Question: {input}
{agent_scratchpad}
""")

researcher_agent = create_react_agent(llm, [web_search], researcher_prompt)
researcher_executor = AgentExecutor(
    agent=researcher_agent, tools=[web_search], verbose=True,
    max_iterations=5, handle_parsing_errors=True   # bumped from 3 to 5
)

def researcher_node(state: AgentState) -> Command[Literal["supervisor"]]:
    question = state["original_question"]
    feedback = state.get("critic_feedback", "")
    input_text = question if not feedback else f"{question}\n\nPrevious feedback to address: {feedback}"

    result = researcher_executor.invoke({"input": input_text})
    output = result["output"]

    # NEW: detect researcher failure before passing it downstream
    if "iteration limit" in output.lower() or "time limit" in output.lower():
        attempts = state.get("research_attempts", 0) + 1
        if attempts >= 2:
            # give up gracefully — tell writer honestly that research was inconclusive
            output = "Unable to find specific, verified information on this topic after multiple attempts."
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=f"[Research findings]: {output}")],
                "research_done": True,
                "research_approved": attempts >= 2,  # skip critic if we're already giving up
                "research_attempts": attempts
            }
        )

    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=f"[Research findings]: {output}")],
            "research_done": True,
            "research_approved": False,
            "research_attempts": state.get("research_attempts", 0) + 1
        }
    )

# =============================================
# CRITIC
# =============================================
critic_prompt = PromptTemplate.from_template("""
You are reviewing research findings for accuracy and completeness.
You can use web_search to verify any claim that seems uncertain or unverified.

Tools available: {tools}
Tool names: {tool_names}

Use this format:
Question: the research to review
Thought: think about whether the research is complete and accurate, verify if needed
Action: the tool to use (only if verification needed)
Action Input: the search query
Observation: the result
... (repeat if needed)
Thought: I have decided
Final Answer: respond with EXACTLY "APPROVED" if the research is sufficient and accurate,
OR "NEEDS_MORE_RESEARCH: <specific feedback on what's missing or wrong>" if not.

Question: {input}
{agent_scratchpad}
""")

critic_agent = create_react_agent(llm, [web_search], critic_prompt)
critic_executor = AgentExecutor(
    agent=critic_agent, tools=[web_search], verbose=True,
    max_iterations=5, handle_parsing_errors=True
)

def critic_node(state: AgentState) -> Command[Literal["supervisor"]]:
    if state.get("research_attempts", 0) >= 2:
        return Command(goto="supervisor", update={"research_approved": True})

    research_text = state["messages"][-1].content
    review_input = f"Original question: {state['original_question']}\n\nResearch findings: {research_text}"

    result = critic_executor.invoke({"input": review_input})
    verdict = result["output"]

    if "iteration limit" in verdict.lower() or "time limit" in verdict.lower():
        return Command(goto="supervisor", update={"research_approved": True})

    if "approved" in verdict.lower():
        return Command(goto="supervisor", update={"research_approved": True})
    else:
        feedback = verdict.replace("NEEDS_MORE_RESEARCH:", "").strip()
        return Command(
            goto="supervisor",
            update={
                "research_approved": False,
                "research_done": False,
                "critic_feedback": feedback
            }
        )

# =============================================
# WRITER
# =============================================
def writer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    system_prompt = "You are a writer. Using the research findings in the conversation, write a clear final answer for the user's original question."
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=response.content)],
            "final_answer_given": True
        }
    )

# =============================================
# BUILD GRAPH (cached so it's not rebuilt every rerun)
# =============================================
@st.cache_resource
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)
    graph.set_entry_point("supervisor")
    return graph.compile()

app = build_graph()

# =============================================
# STREAMLIT UI
# =============================================
st.title("🤖 AI Research Assistant")
st.caption("Researcher → Critic → Writer multi-agent system")

question = st.text_input("Ask a question:", placeholder="What are the latest developments in quantum computing?")

status_labels = {
    "supervisor": "🧭 Deciding next step...",
    "researcher": "🔍 Researching...",
    "critic": "🧐 Reviewing research...",
    "writer": "✍️ Writing final answer..."
}

if st.button("Run Research", type="primary") and question:
    all_messages = [HumanMessage(content=question)]

    with st.status("🧭 Starting...", expanded=True) as status:
        initial_state = {
            "messages": all_messages,
            "original_question": question,
            "loop_count": 0,
            "research_done": False,
            "research_attempts": 0,
            "research_approved": False,
            "critic_feedback": "",
            "final_answer_given": False
        }

        for chunk in app.stream(initial_state, stream_mode="updates"):
            node_name = list(chunk.keys())[0]
            node_update = chunk[node_name]
            label = status_labels.get(node_name, f"Running {node_name}...")
            status.update(label=label)
            st.write(label)
            if node_update and "messages" in node_update:   # <-- added node_update check
                all_messages.extend(node_update["messages"])

        status.update(label="✅ Done!", state="complete")

    st.subheader("Final Answer")
    st.write(all_messages[-1].content)

    with st.expander("See research findings"):
        for msg in all_messages[:-1]:
            st.write(f"**{msg.type}:** {msg.content}")