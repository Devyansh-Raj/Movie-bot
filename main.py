import os
import uuid
import sys
from src.agent_graph import graph
from langchain_core.messages import HumanMessage, AIMessage

def main():
    print("🎬 Movie Maven CLI Chatbot")
    print("Ask me anything about movies! (Type 'quit' or 'exit' to stop)")
    print("-" * 50)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize chat history in graph memory automatically
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
            
        if not user_input:
            continue

        print("Bot: ", end="", flush=True)
        
        try:
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            # Stream the response
            full_response = ""
            for event in graph.stream(inputs, config=config, stream_mode="values"):
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    # Check if it's an AI message to print (avoid printing intermediate tool calls if not desired)
                    # The graph returns the state. We want to print the incremental updates or just final?
                    # "values" mode returns the full state at each step.
                    # We just want the final answer really, or we want to stream tokens if possible.
                    # Since we aren't streaming tokens from the LLM directly in this setup easily without callbacks,
                    # we will just print the final response.
                    pass
            
            # Get final state
            # snap = graph.get_state(config)
            # last_message = snap.values["messages"][-1]
            # Actually, `event` in the loop eventually holds the final state.
            
            # Let's just run invoke for simplicity in CLI if not streaming tokens
            # Or use the last event from the loop
            if isinstance(last_msg, AIMessage):
                print(last_msg.content)
            
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
