from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os
import requests

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# =============================================
# TOOL 1 - Weather Tool
# =============================================
@tool
def weather(city: str) -> str:
    """Get current weather for any city. Input should be just the city name like 'Hyderabad' or 'Mumbai'."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_data = requests.get(geo_url).json()

        if not geo_data.get("results"):
            return f"City '{city}' not found"

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        city_name = geo_data["results"][0]["name"]
        country = geo_data["results"][0]["country"]

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_data = requests.get(weather_url).json()

        temp = weather_data["current_weather"]["temperature"]
        windspeed = weather_data["current_weather"]["windspeed"]

        return f"Weather in {city_name}, {country}: Temperature {temp}°C, Wind speed {windspeed} km/h"

    except Exception as e:
        return f"Error: {str(e)}"

# =============================================
# TOOL 2 - File Reader
# =============================================
@tool
def file_reader(filepath: str) -> str:
    """Read contents of any text file. Input should be the file path like 'notes.txt'."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File '{filepath}' not found"
    except Exception as e:
        return f"Error: {str(e)}"

# =============================================
# TOOL 3 - Wikipedia
# =============================================
@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information about any topic. Input should be the search term."""
    try:
        import wikipedia
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"

# =============================================
# TOOL 4 - Save Notes
# =============================================
@tool
def save_notes(input: str) -> str:
    """Save notes to a file. Input format must be exactly: 'filename.txt|||content to save'"""
    try:
        parts = input.split("|||")
        if len(parts) != 2:
            return "Error: Input must be in format 'filename.txt|||content'"
        filename = parts[0].strip()
        content = parts[1].strip()
        with open(filename, "w") as f:
            f.write(content)
        return f"Successfully saved to {filename}"
    except Exception as e:
        return f"Error saving: {str(e)}"

# =============================================
# TOOL 5 - Web Search
# =============================================
search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the internet for current news and real time information."""
    return search.run(query)

# =============================================
# ALL TOOLS
# =============================================
tools = [weather, file_reader, wikipedia_search, save_notes, web_search]

# =============================================
# PROMPT
# =============================================
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

# =============================================
# AGENT
# =============================================
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=7,
    handle_parsing_errors=True
)

# =============================================
# TESTS
# =============================================
print("=" * 50)
print("TEST 1: Weather")
print("=" * 50)
result1 = agent_executor.invoke({"input": "What is the weather in Hyderabad?"})
print(f"Final Answer: {result1['output']}")

print("\n" + "=" * 50)
print("TEST 2: Wikipedia")
print("=" * 50)
result2 = agent_executor.invoke({"input": "Tell me about Hyderabad city from Wikipedia"})
print(f"Final Answer: {result2['output']}")

print("\n" + "=" * 50)
print("TEST 3: Save Notes")
print("=" * 50)
result3 = agent_executor.invoke({
    "input": "Search Wikipedia about Python programming language and save summary to 'python_notes.txt'"
})
print(f"Final Answer: {result3['output']}")

print("\n" + "=" * 50)
print("TEST 4: Multi Tool")
print("=" * 50)
result4 = agent_executor.invoke({
    "input": "Get weather in Mumbai and search web for latest AI news. Save both results to 'daily_report.txt'"
})
print(f"Final Answer: {result4['output']}")