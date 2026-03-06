import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model_name = "gemini-3-flash-preview"
print(f"Testing model: {model_name}")

if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found.")
    exit(1)

llm = ChatGoogleGenerativeAI(model=model_name)

msg = HumanMessage(content="Hello, are you working?")

print("Invoking LLM without tools...")
try:
    response = llm.invoke([msg])
    print(f"Response: {response.content}")
except Exception as e:
    print(f"Error without tools: {e}")

print("\nInvoking LLM WITH tools...")
from langchain_core.tools import tool

@tool
def dummy_tool(x: int):
    """dummy"""
    return x + 1

llm_with_tools = llm.bind_tools([dummy_tool])

try:
    response = llm_with_tools.invoke([msg])
    print(f"Response with tools: {response.content}")
except Exception as e:
    print(f"Error with tools: {e}")
