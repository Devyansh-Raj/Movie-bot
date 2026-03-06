import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

model_name = "gemini-3-flash-preview"
print(f"Testing model: {model_name} with REAL tools schema")

if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found.")
    exit(1)

@tool
def lookup_movie_details(title: str, attribute: str = None) -> str:
    """
    Useful for finding specific details about a movie when the title is known.
    Args:
        title: The exact or partial title of the movie.
        attribute: Optional. The specific attribute to retrieve. 
    """
    return "details"

@tool
def semantic_search_movies(query: str) -> str:
    """
    Useful for finding movies based on plot, description, theme, or vague queries.
    """
    return "results"

tools = [lookup_movie_details, semantic_search_movies]

llm = ChatGoogleGenerativeAI(model=model_name)
llm_with_tools = llm.bind_tools(tools)

msg = HumanMessage(content="Who directed Toy Story?")

try:
    print("Invoking with real tools schema...")
    response = llm_with_tools.invoke([msg])
    print(f"Response: {response.content}")
    print(f"Tool Calls: {response.tool_calls}")
except Exception as e:
    print(f"FAILED: {e}")
