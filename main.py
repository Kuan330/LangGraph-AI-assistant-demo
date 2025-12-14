# Setting the foundation for an agent that can plan, research, and respond to user queries.
import operator
import os
from typing import Annotated, List, TypedDict

# This loads the .env file so Python can see the keys
from dotenv import load_dotenv
load_dotenv()

# We need these for the Graph logic
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# configurations
# Set to True to pay for OpenAI/Tavily. 
# Set to False to use Free Gemini/DuckDuckGo.
use_paid_version = False 

# First, define the structure of the agent's state.
# Think of this as the "Shared Notebook" that gets passed between workers.
class AgentState(TypedDict):
    topic: str                # The user's input (e.g., "AI in 2025")
    plan: List[str]           # The Planner writes steps here
    research_data: List[str]  # The Researcher adds notes here
    final_answer: str         # The Responder writes the final essay here

    # 'messages' is required by LangGraph to track chat history
    messages: Annotated[List[BaseMessage], operator.add] #checkbox

print("First Step Completed-> Initialising memory.")


# Step 2
# Now, Here is the model and tool setup based on the configuration.

if use_paid_version:
    # paid options (OpenAI + Tavily)
    from langchain_openai import ChatOpenAI
    from langchain_community.tools.tavily_search import TavilySearchResults

    # The Brain
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # The Tool
    web_search_tool = TavilySearchResults(max_results=2)
    print("Using Paid option (OpenAI + Tavily)") #checkbox

else:
    # free options (Gemini + DuckDuckGo)
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.tools import DuckDuckGoSearchRun

    # The large language model (this need to be checked + updated regularly)
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0)
    
    # The Tool (DuckDuckGo)
    # A "Wrapper" so it behaves like Tavily (returning a list)
    class DuckDuckGoWrapper:
        def invoke(self, query):
            search = DuckDuckGoSearchRun()
            result = search.invoke(query)
            return [{"content": result}] # Return as a list to match Tavily
            
    web_search_tool = DuckDuckGoWrapper()
    print("Using Free option (Gemini + DuckDuckGo)") # checkbox


# Step 3
# 3 Nodes Definitions (workers)
def planner_node(state: AgentState):
    print(f"This is the Planner, now planning research for '{state['topic']}'.") # checkbox
    
    prompt = (
        f"You are a research planner. Break down the topic '{state['topic']}' "
        f"into 3 distinct, actionable web search queries. "
        f"Return ONLY the queries as a bulleted list (no intro text)."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # To Check and Fix the output format from Google Gemini
    # Sometimes Google returns a string, sometimes a list with a signature
    # This checks which one it is and extract only the text
    # Handle Google's list format (Text + Signature)
    content = response.content
    if isinstance(content, list):

        # Extract just the text part
        text_content = content[0].get('text', str(content))
    else:
        text_content = str(content)

    # Now the string is clean, split into lines and extract the plan steps
    plan_steps = [line.strip("- *") for line in text_content.split("\n") if line.strip()]
    
    return {"plan": plan_steps}


def researcher_node(state: AgentState):
    print("This is the Researcher, now conducting web searches based on the plan.") # checkbox
    plan = state.get("plan", [])
    research_results = []
    
    for step in plan:
        print(f"Searching for: {step}")
        try:
            # Use the tool that was set up in Step 2
            results = web_search_tool.invoke(step)
            content = results[0]['content']
            research_results.append(f"Source ({step}): {content}")
        except Exception as e:
            research_results.append(f"Error searching {step}: {e}")

    return {"research_data": research_results}


def responder_node(state: AgentState):
    print("This is the Responder, now compiling the final answer.") # checkbox

    # For bug fix: Check if the researcher actually found anything
    if not state["research_data"]:
        print(" DEBUG: Research data is EMPTY! The researcher failed.")
        return {"final_answer": "Error: No research data found. Please try again."}

    data = "\n".join(state["research_data"])
    
    prompt = (
        f"Topic: {state['topic']}\n"
        f"Research Data:\n{data}\n\n"
        f"Write a professional summary based ONLY on the data above."
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])

    # Bug fix 2: See exactly what Google sent back (even if it's hidden)
    print(f" DEBUG RAW RESPONSE: {response}")

    # To Check and Fix the output format from Google Gemini
    # Sometimes Google returns a string, sometimes a list with a signature.
    # This checks which one it is and extract only the text.
    content = response.content
    
    if isinstance(content, list):
        # If it's a list, grab the text from the first part
        # content looks like: [{'text': 'The answer...', 'type': 'text'}]
        final_text = content[0].get('text', str(content))
    else:
        # If it's already a string, just use it
        final_text = content

        
    return {"final_answer": final_text}


# Step 4
# Now, set up the StateGraph to connect everything together.
workflow = StateGraph(AgentState)

# Add the workers to the graph
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("responder", responder_node)

# Flow definition - the order in which nodes are executed
workflow.set_entry_point("planner")        # Start with the Planner
workflow.add_edge("planner", "researcher") # Then go to Researcher
workflow.add_edge("researcher", "responder") # Then to Responder
workflow.add_edge("responder", END)        # Then finish

# Compile the graph into an executable app
app = workflow.compile()


# Step 5
# Here test the entire setup with a sample topic.
if __name__ == "__main__":
    # Here to change the research topic
    topic = "\n The current state of AI computing in 2025"

    print(f"\n Starting the research based on: {topic}\n")

    # Run the graph
    result = app.invoke({"topic": topic, "messages": []})
    
    print("\n" + "="*40)
    print(" Here is the final answer:\n")
    print("="*40)
    print(result["final_answer"])