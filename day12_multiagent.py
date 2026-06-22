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

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    original_question: str
    loop_count: int
    research_done: bool
    final_answer_given: bool

def supervisor_node(state: AgentState) -> Command[Literal["researcher", "writer", "__end__"]]:
    if state.get("final_answer_given"):
        print("\n🛑 Final answer already given — finishing.")
        return Command(goto=END)

    if state.get("loop_count", 0) >= 4:
        print("\n🛑 Loop limit hit — forcing finish.")
        return Command(goto=END)

    if not state.get("research_done"):
        print("\n🧭 Supervisor: research not done yet → researcher")
        return Command(goto="researcher", update={"loop_count": state.get("loop_count", 0) + 1})

    print("\n🧭 Supervisor: research done → writer")
    return Command(goto="writer", update={"loop_count": state.get("loop_count", 0) + 1})

researcher_prompt = PromptTemplate.from_template("""
Answer the question using the web_search tool to find current facts.

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
    agent=researcher_agent,
    tools=[web_search],
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True
)

def researcher_node(state: AgentState) -> Command[Literal["supervisor"]]:
    question = state["original_question"]
    result = researcher_executor.invoke({"input": question})
    print(f"\n🔍 Researcher found: {result['output']}")
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=f"[Research findings]: {result['output']}")],
            "research_done": True
        }
    )

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

graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.set_entry_point("supervisor")
app = graph.compile()

question = "What are the latest developments in quantum computing in 2026?"
result = app.invoke({
    "messages": [HumanMessage(content=question)],
    "original_question": question,
    "loop_count": 0,
    "research_done": False,
    "final_answer_given": False
})

print("\n" + "="*50)
print("FINAL CONVERSATION:")
print("="*50)
for msg in result["messages"]:
    print(f"\n[{msg.type}]: {msg.content}")