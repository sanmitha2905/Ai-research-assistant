from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1 - Setup LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Step 2 - Define Tools
@tool
def calculator(expression: str) -> str:
    """Use this tool to calculate any math expression. Input should be a math expression like '2 + 2' or '10 * 5'."""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Error: Invalid math expression"

@tool
def word_counter(text: str) -> str:
    """Use this tool to count the number of words in a text."""
    count = len(text.split())
    return f"The text has {count} words"

@tool
def reverse_text(text: str) -> str:
    """Use this tool to reverse any text."""
    return text[::-1]
from langchain_community.tools import DuckDuckGoSearchRun

# Web Search Tool
search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Use this tool to search the internet for current information, news, or any real time data."""
    result = search.run(query)
    return result
# Step 3 - List of tools
tools = [calculator, word_counter, reverse_text, web_search]

# Step 4 - ReAct Prompt
prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

# Step 5 - Create Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Step 6 - Create Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# Step 7 - Run Agent
print("=" * 50)
print("TEST 1: Math calculation")
print("=" * 50)
result = agent_executor.invoke({
    "input": "What is 25 multiplied by 48?"
})
print(f"Final Answer: {result['output']}")

print("\n" + "=" * 50)
print("TEST 2: Word count")
print("=" * 50)
result2 = agent_executor.invoke({
    "input": "How many words are in this sentence: I am Sanmi and I am learning Agentic AI"
})
print(f"Final Answer: {result2['output']}")

print("\n" + "=" * 50)
print("TEST 3: Agent decides which tool")
print("=" * 50)
result3 = agent_executor.invoke({
    "input": "Reverse the word 'Hyderabad' and then count how many characters it has after reversing"
})
print(f"Final Answer: {result3['output']}")
print("\n" + "=" * 50)
print("TEST 4: Web Search")
print("=" * 50)
result4 = agent_executor.invoke({
    "input": "What are the latest developments in AI in 2025?"
})
print(f"Final Answer: {result4['output']}")