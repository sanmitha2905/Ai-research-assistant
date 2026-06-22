from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# =============================================
# STEP 1 - State (FIXED: Annotated + add_messages)
# =============================================
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# =============================================
# STEP 2 - Nodes
# =============================================
def chat_node(state: AgentState) -> AgentState:
    print(f"\n💭 Thinking...")
    response = llm.invoke(state["messages"])
    print(f"✅ Answer: {response.content}")
    return {"messages": [response]}

# =============================================
# STEP 3 - Build Graph
# =============================================
graph = StateGraph(AgentState)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.add_edge("chat", END)

# =============================================
# STEP 4 - MemorySaver (Checkpointer)
# =============================================
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

print("✅ Graph with MemorySaver built!")

# SESSION 1 — Sanmi
print("\n" + "=" * 50)
print("SESSION 1 — Sanmi")
print("=" * 50)

sanmi_config = {"configurable": {"thread_id": "sanmi_session"}}

print("\n--- Turn 1 ---")
app.invoke({"messages": [HumanMessage(content="My name is Sanmi and I am from Hyderabad")]}, config=sanmi_config)

print("\n--- Turn 2 ---")
app.invoke({"messages": [HumanMessage(content="What is my name?")]}, config=sanmi_config)

print("\n--- Turn 3 ---")
app.invoke({"messages": [HumanMessage(content="Where am I from?")]}, config=sanmi_config)

# SESSION 2 — John
print("\n" + "=" * 50)
print("SESSION 2 — Different User")
print("=" * 50)

john_config = {"configurable": {"thread_id": "john_session"}}

print("\n--- Turn 1 ---")
app.invoke({"messages": [HumanMessage(content="My name is John and I am from New York")]}, config=john_config)

print("\n--- Turn 2 ---")
app.invoke({"messages": [HumanMessage(content="What is my name?")]}, config=john_config)

# BACK TO SESSION 1
print("\n" + "=" * 50)
print("BACK TO SESSION 1 — Sanmi")
print("=" * 50)

print("\n--- Does it still remember Sanmi? ---")
app.invoke({"messages": [HumanMessage(content="What is my name and where am I from?")]}, config=sanmi_config)

# =============================================
# STEP 6 - SQLite Checkpointer
# =============================================
print("\n" + "=" * 50)
print("SQLITE CHECKPOINTER — Persistent Memory")
print("=" * 50)

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("memory.db", check_same_thread=False)
sqlite_checkpointer = SqliteSaver(conn)

app_persistent = graph.compile(checkpointer=sqlite_checkpointer)

persistent_config = {"configurable": {"thread_id": "persistent_session"}}

print("\n--- Saving to database ---")
app_persistent.invoke(
    {"messages": [HumanMessage(content="Remember this: My favorite color is blue and I love AI")]},
    config=persistent_config
)

print("\n--- Retrieving from database ---")
app_persistent.invoke(
    {"messages": [HumanMessage(content="What is my favorite color?")]},
    config=persistent_config
)

print("\n✅ Memory saved to memory.db file!")

print("\n" + "=" * 50)
print("WHAT IS STORED IN memory.db:")
print("=" * 50)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(f"Tables in database: {cursor.fetchall()}")

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, List, Annotated
import sqlite3

llm = ChatGroq(model="llama-3.3-70b-versatile")

class SummaryState(TypedDict):
     messages: Annotated[List, add_messages]
     summary: str

def chatbot_node(state: SummaryState):
    # build context: summary (if any) + recent messages
    context = []
    if state.get("summary"):
        context.append(SystemMessage(content=f"Summary of earlier conversation: {state['summary']}"))
    context.extend(state["messages"])

    response = llm.invoke(context)
    return {"messages": [response]}

def should_summarize(state: SummaryState):
    if len(state["messages"]) > 10:
        return "summarize"
    return "end"

def summarize_node(state: SummaryState):
    old_messages = state["messages"][:-4]   # keep last 4 fresh, summarize the rest
    recent_messages = state["messages"][-4:]

    text_to_summarize = "\n".join([f"{m.type}: {m.content}" for m in old_messages])
    prompt = f"Summarize this conversation briefly, keeping key facts:\n{text_to_summarize}"

    summary_response = llm.invoke([HumanMessage(content=prompt)])
    existing_summary = state.get("summary", "")
    new_summary = existing_summary + "\n" + summary_response.content

    return {"messages": recent_messages, "summary": new_summary}

graph = StateGraph(SummaryState)
graph.add_node("chatbot", chatbot_node)
graph.add_node("summarize", summarize_node)

graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot", should_summarize, {
    "summarize": "summarize",
    "end": END
})
graph.add_edge("summarize", END)

conn = sqlite3.connect("memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user1"}}

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    print("Bot:", result["messages"][-1].content)
    if result.get("summary"):
        print("[Summary so far]:", result["summary"])