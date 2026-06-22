from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, List
import os

load_dotenv()

# =============================================
# STEP 1 - Define State
# =============================================
# TypedDict defines exactly what our state contains
class AgentState(TypedDict):
    messages: List        # full conversation history
    current_input: str    # what user just asked
    search_result: str    # result from any search
    final_answer: str     # agent's final answer
    step_count: int       # how many steps taken

# =============================================
# STEP 2 - Setup LLM
# =============================================
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# =============================================
# STEP 3 - Define Nodes
# Each node = one function = one job
# =============================================

# Node 1 - Greet user and understand question
def understand_node(state: AgentState) -> AgentState:
    print(f"\n🧠 UNDERSTAND NODE — Processing: '{state['current_input']}'")
    
    # Add user message to history
    state["messages"].append(
        HumanMessage(content=state["current_input"])
    )
    state["step_count"] += 1
    print(f"   Step count: {state['step_count']}")
    return state

# Node 2 - Think and generate answer using full history
def think_node(state: AgentState) -> AgentState:
    print(f"\n💭 THINK NODE — Generating answer with memory...")
    
    # Send FULL message history to LLM
    # This is how memory works — LLM sees everything!
    response = llm.invoke(state["messages"])
    
    state["final_answer"] = response.content
    state["step_count"] += 1
    print(f"   Step count: {state['step_count']}")
    return state

# Node 3 - Save answer to history and display
def respond_node(state: AgentState) -> AgentState:
    print(f"\n✅ RESPOND NODE — Saving answer to memory...")
    
    # Add AI answer to history
    # Next question will see this answer too!
    state["messages"].append(
        AIMessage(content=state["final_answer"])
    )
    state["step_count"] += 1
    print(f"   Final Answer: {state['final_answer']}")
    print(f"   Total messages in memory: {len(state['messages'])}")
    return state

# =============================================
# STEP 4 - Build the Graph
# =============================================
print("Building LangGraph...")

# Create graph with our state
graph = StateGraph(AgentState)

# Add nodes to graph
graph.add_node("understand", understand_node)
graph.add_node("think", think_node)
graph.add_node("respond", respond_node)

# Add edges — connect nodes in order
graph.add_edge("understand", "think")
graph.add_edge("think", "respond")
graph.add_edge("respond", END)

# Set entry point — where graph starts
graph.set_entry_point("understand")

# Compile graph — makes it ready to run
app = graph.compile()

print("✅ Graph built successfully!")
print("Nodes: understand → think → respond → END")

# =============================================
# STEP 5 - Initialize State
# =============================================
state = {
    "messages": [],
    "current_input": "",
    "search_result": "",
    "final_answer": "",
    "step_count": 0
}

# =============================================
# STEP 6 - Have a conversation WITH MEMORY
# =============================================
print("\n" + "=" * 50)
print("CONVERSATION WITH MEMORY")
print("=" * 50)

# Conversation 1
print("\n--- Turn 1 ---")
state["current_input"] = "My name is Sanmi and I am learning Agentic AI"
state = app.invoke(state)
print(f"Answer: {state['final_answer']}")

# Conversation 2 — does it remember?
print("\n--- Turn 2 ---")
state["current_input"] = "What is my name?"
state = app.invoke(state)
print(f"Answer: {state['final_answer']}")

# Conversation 3 — does it remember more?
print("\n--- Turn 3 ---")
state["current_input"] = "What am I learning?"
state = app.invoke(state)
print(f"Answer: {state['final_answer']}")

# Conversation 4 — complex question using memory
print("\n--- Turn 4 ---")
state["current_input"] = "Based on what I told you, give me one tip for my learning journey"
state = app.invoke(state)
print(f"Answer: {state['final_answer']}")

# Show full memory at end
print("\n" + "=" * 50)
print("FULL MEMORY CONTENTS:")
print("=" * 50)
for i, msg in enumerate(state["messages"]):
    role = "YOU" if isinstance(msg, HumanMessage) else "AI"
    print(f"\n{role} [{i+1}]: {msg.content[:100]}...")

print(f"\nTotal steps taken: {state['step_count']}")
print(f"Total messages in memory: {len(state['messages'])}")

# =============================================
# PART 2 - CONDITIONAL EDGES
# =============================================
print("\n" + "=" * 50)
print("PART 2: CONDITIONAL EDGES")
print("=" * 50)

from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

# New state for conditional graph
class SmartAgentState(TypedDict):
    messages: List
    current_input: str
    needs_search: bool      # NEW — does question need web search?
    search_result: str
    final_answer: str

# Node 1 - Decide if search is needed
def decide_node(state: SmartAgentState) -> SmartAgentState:
    print(f"\n🤔 DECIDE NODE — Question: '{state['current_input']}'")
    
    question = state["current_input"].lower()
    
    # Simple rule — if question has these words, search web
    search_keywords = ["latest", "today", "news", "current", 
                      "weather", "2025", "2026", "recent"]
    
    needs_search = any(word in question for word in search_keywords)
    state["needs_search"] = needs_search
    
    if needs_search:
        print("   Decision: NEEDS WEB SEARCH 🔍")
    else:
        print("   Decision: CAN ANSWER FROM MEMORY 🧠")
    
    return state

# Node 2 - Search web
def search_node(state: SmartAgentState) -> SmartAgentState:
    print(f"\n🔍 SEARCH NODE — Searching web...")
    result = search.run(state["current_input"])
    state["search_result"] = result
    print(f"   Got search result: {result[:100]}...")
    return state

# Node 3 - Answer from memory only
def memory_answer_node(state: SmartAgentState) -> SmartAgentState:
    print(f"\n🧠 MEMORY ANSWER NODE — Answering from history...")
    
    messages = state["messages"] + [
        HumanMessage(content=state["current_input"])
    ]
    response = llm.invoke(messages)
    state["final_answer"] = response.content
    state["messages"].append(HumanMessage(content=state["current_input"]))
    state["messages"].append(AIMessage(content=state["final_answer"]))
    return state

# Node 4 - Answer using search result
def search_answer_node(state: SmartAgentState) -> SmartAgentState:
    print(f"\n💡 SEARCH ANSWER NODE — Combining search + memory...")
    
    # Give LLM both the search result AND conversation history
    prompt = f"""Based on this search result:
{state['search_result']}

Answer this question: {state['current_input']}
Be concise and clear."""
    
    messages = state["messages"] + [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    state["final_answer"] = response.content
    state["messages"].append(HumanMessage(content=state["current_input"]))
    state["messages"].append(AIMessage(content=state["final_answer"]))
    return state

# Condition function — this decides which node to go to
def should_search(state: SmartAgentState) -> str:
    if state["needs_search"]:
        return "search"      # go to search_node
    else:
        return "memory"      # go to memory_answer_node

# Build smart graph
smart_graph = StateGraph(SmartAgentState)

# Add nodes
smart_graph.add_node("decide", decide_node)
smart_graph.add_node("search", search_node)
smart_graph.add_node("memory_answer", memory_answer_node)
smart_graph.add_node("search_answer", search_answer_node)

# Set entry point
smart_graph.set_entry_point("decide")

# CONDITIONAL EDGE — key part!
smart_graph.add_conditional_edges(
    "decide",           # from this node
    should_search,      # run this function to decide
    {
        "search": "search",           # if returns "search" → go to search node
        "memory": "memory_answer"     # if returns "memory" → go to memory_answer node
    }
)

# After search → go to search_answer
smart_graph.add_edge("search", "search_answer")

# Both answer nodes → END
smart_graph.add_edge("memory_answer", END)
smart_graph.add_edge("search_answer", END)

# Compile
smart_app = smart_graph.compile()

print("✅ Smart Graph built!")
print("""
Graph structure:
                    → search_node → search_answer_node → END
decide_node →
                    → memory_answer_node → END
""")

# Test smart graph
smart_state = {
    "messages": [],
    "current_input": "",
    "needs_search": False,
    "search_result": "",
    "final_answer": ""
}

# Question 1 - needs search
print("\n--- Smart Test 1: Needs Search ---")
smart_state["current_input"] = "What is the latest AI news today?"
smart_state = smart_app.invoke(smart_state)
print(f"Answer: {smart_state['final_answer'][:200]}...")

# Question 2 - from memory
print("\n--- Smart Test 2: From Memory ---")
smart_state["current_input"] = "What is LangGraph?"
smart_state = smart_app.invoke(smart_state)
print(f"Answer: {smart_state['final_answer'][:200]}...")

# Question 3 - needs search
print("\n--- Smart Test 3: Needs Search ---")
smart_state["current_input"] = "What is the current weather in Hyderabad?"
smart_state = smart_app.invoke(smart_state)
print(f"Answer: {smart_state['final_answer'][:200]}...")