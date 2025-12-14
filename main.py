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

    # The Brain (Gemini 1.5 Flash is free-tier eligible)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
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
    print(f"This is the Planner, now planning research for '{state['topic']}'") # checkbox
    
    # Ask the LLM to generate 3 search queries
    prompt = (
        f"You are a research planner. Break down the topic '{state['topic']}' "
        f"into 3 distinct, actionable web search queries. "
        f"Return ONLY the queries as a bulleted list (no intro text)."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Clean up the response to get a python list
    # Split by newlines and remove bullet points
    plan_steps = [line.strip("- *") for line in response.content.split("\n") if line.strip()]
    
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
    # Combine all research notes into one big string
    data = "\n".join(state["research_data"])
    
    prompt = (
        f"Topic: {state['topic']}\n"
        f"Research Data:\n{data}\n\n"
        f"Write a professional summary based ONLY on the data above."
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_answer": response.content}