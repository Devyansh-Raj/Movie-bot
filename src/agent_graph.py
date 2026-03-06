import os
import pandas as pd
import numpy as np
from typing import Annotated, Literal, TypedDict, Union
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()

# Load Data (Global for cache)
# Adjust path as needed based on where this is run
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "processed_movies.pkl")
INDICES_DIR = os.path.join(BASE_DIR, "indices")

print(f"Loading data from {PROCESSED_DATA_PATH}...")
try:
    df = pd.read_pickle(PROCESSED_DATA_PATH)
    print("Data loaded.")
except FileNotFoundError:
    print("Processed data not found. Please run data_processor.py first.")
    df = pd.DataFrame() # Fallback

# Initialize Vector Store
print("Loading FAISS index...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        os.path.join(INDICES_DIR, "movies_faiss_index"), 
        embeddings,
        allow_dangerous_deserialization=True 
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("FAISS index loaded.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    retriever = None

# --- Toools ---

@tool
def lookup_movie_details(title: str, attribute: str = None) -> str:
    """
    Useful for finding specific details about a movie when the title is known.
    Args:
        title: The exact or partial title of the movie.
        attribute: Optional. The specific attribute to retrieve (e.g., 'revenue', 'budget', 'release_date', 'overview', 'cast', 'director', 'production_countries', 'tagline'). 
                   If None, returns a summary.
    """
    # Simple fuzzy match
    # In production, use elastic search or similar. Here we use pandas str.contains
    if df.empty:
        return "Data not loaded."
    
    # Case insensitive match
    matches = df[df['title'].str.contains(title, case=False, na=False)]
    
    if matches.empty:
        return f"No movie found matching '{title}'."
    
    if len(matches) > 1:
        # Return top match or list
        # For chat, usually return the most popular one or clarify
        # heuristic: sort by popularity or vote_count
        matches = matches.sort_values('vote_count', ascending=False)
        best_match = matches.iloc[0]
        # Maybe mention others?
        other_titles = matches['title'].unique()[:3]
        if len(other_titles) > 1:
             response = f"Found multiple movies: {', '.join(other_titles)}. Using '{best_match['title']}' ({best_match['release_date']}).\n"
        else:
            response = ""
    else:
        best_match = matches.iloc[0]
        response = ""

    if attribute:
        attr = attribute.lower()
        # Map common terms
        if 'country' in attr or 'countries' in attr:
            attr = 'production_countries'
        
        if attr in best_match:
            val = best_match[attr]
            if isinstance(val, list):
                val = ', '.join(val)
            return f"{response}{attribute} of {best_match['title']}: {val}"
        elif 'box office' in attr or 'money' in attr:
            return f"{response}{best_match['title']} Revenue: ${best_match['revenue']:,.2f}, Budget: ${best_match['budget']:,.2f}"
    
    # Default Summary
    summary = (
        f"Title: {best_match['title']}\n"
        f"Year: {best_match['release_date']}\n"
        f"Genres: {', '.join(best_match['genres'])}\n"
        f"Imdb Rating: {best_match['vote_average']}\n"
        f"Overview: {best_match['overview']}\n"
        f"Director: {best_match['director']}\n"
        f"Cast: {', '.join(best_match['cast'])}\n"
        f"Countries: {', '.join(best_match['production_countries'])}\n"
        f"Tagline: {best_match['tagline']}"
    )
    return response + summary

@tool
def semantic_search_movies(query: str) -> str:
    """
    Useful for finding movies based on plot, description, theme, or vague queries.
    e.g., "movies about time travel", "scary movies with dolls".
    """
    if not retriever:
        return "Vector search unavailable."
    
    docs = retriever.invoke(query)
    results = []
    for doc in docs:
        results.append(f"Title: {doc.metadata['title']} (ID: {doc.metadata['id']})\nSummary: {doc.page_content[:200]}...")
    
    return "\n---\n".join(results)

tools = [lookup_movie_details, semantic_search_movies]

# --- Graph ---

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize LLM
if os.getenv("GOOGLE_API_KEY"):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
else:
    # Fallback or error
    print("WARNING: GOOGLE_API_KEY not found.")
    llm = None

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot",  
                                    lambda state: "tools" if state["messages"][-1].tool_calls else END)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile(checkpointer=MemorySaver())

def get_response(user_input: str, thread_id: str = "1"):
    config = {"configurable": {"thread_id": thread_id}}
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)]}, 
        config, 
        stream_mode="values"
    )
    # Return the last message content
    last_msg = None
    for event in events:
        if "messages" in event:
            last_msg = event["messages"][-1]
    
    return last_msg.content if last_msg else "No response."

if __name__ == "__main__":
    # Test
    print("Testing agent...")
    print(get_response("Tell me about the movie Toy Story."))
    print(get_response("Who directed it?"))
