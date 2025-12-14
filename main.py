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
    messages: Annotated[List[BaseMessage], operator.add]

print("First Step Completed-> Initialising memory.")