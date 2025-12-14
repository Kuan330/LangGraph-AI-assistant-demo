"""
Research Assistant Agent using LangGraph

Description:
    This script implements a LangGraph-based agent capable of planning, 
    researching, and summarizing topics using external search tools.
    
Features:
    - Multi-Node Architecture (Planner -> Researcher -> Responder)
    - Conditional Logic (Auto-retry if search fails)
    - Human-in-the-Loop (Pauses for approval before writing)
    - Persistent State (MemorySaver)
    - PDF Export Generation
"""


# Setting the foundation for an agent that can plan, research, and respond to user queries.
import operator
import os
from datetime import datetime
from typing import Annotated, List, TypedDict, Literal

# Load environment variables
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # this for Pausing
from fpdf import FPDF  #  this is for PDF

# This loads the .env file so Python can see the keys
load_dotenv()

# configurations
# Set to True to pay for OpenAI/Tavily. 
# Set to False to use Free Gemini/DuckDuckGo.
use_paid_version = False 


# Step 1
# First, define the structure of the agent's state.
# Think of this as the "Shared Notebook" that gets passed between workers.
class AgentState(TypedDict):
    """
    The persistent state of the agent, passed between graph nodes.
    """

    topic: str
    plan: List[str]
    research_data: List[str]
    final_answer: str
    messages: Annotated[List[BaseMessage], operator.add]
    loop_count: int  # Required to stop infinite loops in conditional edges

print("System initialised.") 


# Step 2
# Now, this is the model and tool setup based on the configuration.

if use_paid_version:
    # paid options (OpenAI + Tavily)
    from langchain_openai import ChatOpenAI
    from langchain_community.tools.tavily_search import TavilySearchResults

    # The Brain
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # The Tool
    web_search_tool = TavilySearchResults(max_results=2)
    print(" Mode: Paid (OpenAI + Tavily)") 

else:
    # free options (Gemini + DuckDuckGo)
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.tools import DuckDuckGoSearchRun

    # The large language model (this need to be checked + updated regularly)
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0)
    
    # The Tool (DuckDuckGo)
    # A "Wrapper" so it behaves like Tavily (returning a list)
    class DuckDuckGoWrapper:
        """
        Wrapper to standardize DuckDuckGo output to match Tavily's list format.
        """
        def invoke(self, query):
            try:
                search = DuckDuckGoSearchRun()
                result = search.invoke(query)
                return [{"content": result}]
            except Exception as e:
                return [] 
                
    web_search_tool = DuckDuckGoWrapper()
    print(" Mode: Free (Gemini + DuckDuckGo)")


# Time helper function
def get_current_date():
    """
    Tool: Returns the current date as a string (YYYY-MM-DD).
    Used by the Planner to ensure research context is up-to-date.
    """

    return datetime.now().strftime("%Y-%m-%d")


# Step 3
# 3 Nodes Definitions (workers)
def planner_node(state: AgentState):
    """
    Node: Generates a research plan based on the user's topic.
    
    Returns:
        dict: Updates the 'plan' and increments 'loop_count'.
    """

    print(f"This is the Planner, now planning research for '{state['topic']}'.")

    today = get_current_date()

    prompt = (f"Today is {today}. Topic: '{state['topic']}'. "
                "Break this into 5 distincts, actionable web search queries. "
                "Return ONLY the queries as a bulleted list.")
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
    
    # Increment loop count so do not loop forever
    current_loop = state.get("loop_count", 0)
    return {"plan": plan_steps, "loop_count": current_loop + 1}


def researcher_node(state: AgentState):
    """
    Node: Executes the search queries defined in the plan.
    
    Returns:
        dict: Updates 'research_data' with search results.
    """

    print("This is the Researcher, now conducting web searches based on the plan.") # checkbox
    plan = state.get("plan", [])
    research_results = []
    
    for step in plan:
        print(f"Searching for: {step}")
        try:
            results = web_search_tool.invoke(step)

            # Handle empty results for conditional logic
            if not results:
                continue

            # Format normalisation for different tool outputs
            if isinstance(results, list):
                content = results[0].get('content', str(results))
            else:
                content = str(results)
                
            research_results.append(f"Source ({step}): {content}")

        except Exception as e:
            research_results.append(f"Error searching for {step}: {e}")

    return {"research_data": research_results}


def responder_node(state: AgentState):
    """
    Node: Synthesizes the gathered data into a final answer.
    
    Returns:
        dict: Updates 'final_answer'.
    """
    
    print("This is the Responder, now compiling the final answer.") # checkbox

    # For bug fix: Check if the researcher actually found anything
    if not state["research_data"]:
        print(" DEBUG: Research data is EMPTY! The researcher failed.")
        return {"final_answer": "Error: No research data found. Please try again."}

    data = "\n".join(state["research_data"])
    
    prompt = (
            f"Topic: {state['topic']}\n"
            f"Research Data:\n{data}\n\n"
            f"Write a professional summary based ONLY on the data above. "
            f"Please explicitly cite your sources in the text.")
    
    response = llm.invoke([HumanMessage(content=prompt)])

    # Bug fix 2: See exactly what Google sent back (even if it's hidden)
    print(f" Debug raw response: {response}")

    # Format normalisation for different tool outputs
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
# Routing logic based on data quality
def router_logic(state: AgentState) -> Literal["responder", "planner"]:
    """
    Conditional Logic: Decides the next step based on data quality.
    
    Returns:
        'responder': If data was found OR retry limit reached.
        'planner': If no data was found (triggers a retry).
    """

    data_count = len(state.get("research_data", []))
    loop_count = state.get("loop_count", 0)
    
    if data_count > 0 or loop_count >= 2:
        return "responder"
    else:
        print(f" Decision: No data found. Now Retrying... (Attempt {loop_count}) ---")
        return "planner"


# PDF saving function
def save_to_pdf(text, filename="Research_Report.pdf"):
    """
    Utility: Saves the provided text string to a PDF file.
    Includes Latin-1 encoding handling to prevent crashes on special characters.
    """

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Handle Unicode issues 
    text = text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(0, 10, txt=text)
    pdf.output(filename)
    print(f"\n PDF saved successfully: {filename}")



# Step 5
# Now, set up the StateGraph to connect everything together, with pausing support.
workflow = StateGraph(AgentState)

# Add the workers to the graph
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("responder", responder_node)

# Add edges between nodes
workflow.set_entry_point("planner")   
workflow.add_edge("planner", "researcher") 

# Logic to decide next step based on research results
workflow.add_conditional_edges("researcher", router_logic, {"planner": "planner", "responder": "responder"})
workflow.add_edge("responder", END)

# Set up the checkpointer for pausing
memory = MemorySaver() # This saves the state to pause
app = workflow.compile(
    checkpointer=memory, 
    interrupt_before=["responder"]) # Pause before writing the final report


# Step 6
# Here test the entire setup with a sample topic.
if __name__ == "__main__":
    # Here to change the research topic
    topic = "The current state of Nvidia's AI technology and its future prospects."
    print(f"\n Starting the research based on: {topic}\n")

    # Thread ID to track the conversation state
    thread_config = {"configurable": {"thread_id": "1"}}

    # Run the graph until the pause point
    app.invoke({"topic": topic, "messages": [], "loop_count": 0}, config=thread_config)
    
    # Human Approval
    print("\n" + "="*40)
    print("Here pause: The Research is done. Ready to write the report.")
    user_input = input("Now proceed with writing the Final Report? (yes/no): ")

    if user_input.lower() in ["yes", "y"]:
        print("\n Approved. Now writing the Final Report...\n")
            
        # Resume execution (Pass None to continue from the last state)
        result = app.invoke(None, config=thread_config)
        
        final_text = result["final_answer"]
        print("\n" + "="*40)
        print("Final Report")
        print("="*40)
        print(final_text)

        # Save to PDF
        save_to_pdf(final_text)
            
    else:
        print("\n Operation Cancelled.")