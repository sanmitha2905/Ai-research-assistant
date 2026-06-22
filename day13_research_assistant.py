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

@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    return "\n".join([f"{r['title']}: {r['body']}" for r in results])

# =============================================
# STATE — new fields for critic loop
# =============================================
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    original_question: str
    loop_count: int
    research_done: bool
    research_attempts: int        # NEW — caps researcher re-runs
    research_approved: bool       # NEW — critic's verdict
    critic_feedback: str          # NEW — fed back to researcher if rejected
    final_answer_given: bool

# =============================================
# SUPERVISOR — still deterministic
# =============================================
def supervisor_node(state: AgentState) -> Command[Literal["researcher", "critic", "writer", "__end__"]]:
    if state.get("final_answer_given"):
        print("\n🛑 Final answer already given — finishing.")
        return Command(goto=END)

    if state.get("loop_count", 0) >= 6:
        print("\n🛑 Loop limit hit — forcing finish.")
        return Command(goto=END)

    if not state.get("research_done"):
        print("\n🧭 Supervisor: no research yet → researcher")
        return Command(goto="researcher", update={"loop_count": state.get("loop_count", 0) + 1})

    if not state.get("research_approved"):
        print("\n🧭 Supervisor: research not yet reviewed → critic")
        return Command(goto="critic", update={"loop_count": state.get("loop_count", 0) + 1})

    print("\n🧭 Supervisor: research approved → writer")
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
    max_iterations=3, handle_parsing_errors=True
)

def researcher_node(state: AgentState) -> Command[Literal["supervisor"]]:
    question = state["original_question"]
    feedback = state.get("critic_feedback", "")

    input_text = question
    if feedback:
        input_text = f"{question}\n\nPrevious feedback to address: {feedback}"

    result = researcher_executor.invoke({"input": input_text})
    print(f"\n🔍 Researcher found: {result['output']}")

    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=f"[Research findings]: {result['output']}")],
            "research_done": True,
            "research_approved": False,  # reset — needs fresh review
            "research_attempts": state.get("research_attempts", 0) + 1
        }
    )

# =============================================
# CRITIC — real agent with web_search tool
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
    max_iterations=5, handle_parsing_errors=True   # bumped from 3 to 5
)

def critic_node(state: AgentState) -> Command[Literal["supervisor"]]:
    if state.get("research_attempts", 0) >= 2:
        print("\n⚠️ Max research attempts reached — auto-approving to avoid infinite loop.")
        return Command(goto="supervisor", update={"research_approved": True})

    research_text = state["messages"][-1].content
    review_input = f"Original question: {state['original_question']}\n\nResearch findings: {research_text}"

    result = critic_executor.invoke({"input": review_input})
    verdict = result["output"]
    print(f"\n🧐 Critic verdict: {verdict}")

    # NEW: if critic itself failed to produce a real verdict, don't trust it as a rejection
    if "iteration limit" in verdict.lower() or "time limit" in verdict.lower():
        print("\n⚠️ Critic failed to reach a verdict — auto-approving instead of treating as rejection.")
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
    print(f"\n✍️ Writer's final answer: {response.content}")
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=response.content)],
            "final_answer_given": True
        }
    )

# =============================================
# GRAPH
# =============================================
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("critic", critic_node)
graph.add_node("writer", writer_node)
graph.set_entry_point("supervisor")
app = graph.compile()

question = "What are the latest developments in quantum computing in 2026?"
result = app.invoke({
    "messages": [HumanMessage(content=question)],
    "original_question": question,
    "loop_count": 0,
    "research_done": False,
    "research_attempts": 0,
    "research_approved": False,
    "critic_feedback": "",
    "final_answer_given": False
})

print("\n" + "="*50)
print("FINAL CONVERSATION:")
print("="*50)
for msg in result["messages"]:
    print(f"\n[{msg.type}]: {msg.content}")