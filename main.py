"""
This is the main code for the Research Assistant Agent using LangGraph.
It sets up an agent that can plan, research, and respond to user queries
using either free or paid AI services based on configuration.

3 Steps:
1. Define the AgentState structure.
2. Set up the LLM and web search tool based on configuration.
3. Define the planner, researcher, and responder nodes.
4. Set up routing logic and compile the StateGraph into an executable app.
5. Test the entire setup with a sample topic.
6. Run the agent and print the final answer.

"""


# Setting the foundation for an agent that can plan, research, and respond to user queries.
import operator
import os
from datetime import datetime
from typing import Annotated, List, TypedDict, Literal

# Load environment variables
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# This loads the .env file so Python can see the keys
load_dotenv()

# We need these for the Graph logic
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# configurations
# Set to True to pay for OpenAI/Tavily. 
# Set to False to use Free Gemini/DuckDuckGo.
use_paid_version = False 


# Step 1
# First, define the structure of the agent's state.
# Think of this as the "Shared Notebook" that gets passed between workers.
class AgentState(TypedDict):
    topic: str
    plan: List[str]
    research_data: List[str]
    final_answer: str
    messages: Annotated[List[BaseMessage], operator.add]
    loop_count: int  # Required to stop infinite loops in conditional edges

print("First Step Completed-> Initialising memory.") # checkbox


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
                try:
                    search = DuckDuckGoSearchRun()
                    result = search.invoke(query)
                    return [{"content": result}]
                except Exception as e:
                    return [] 
                
    web_search_tool = DuckDuckGoWrapper()


# Time helper function
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

# Step 3
# 3 Nodes Definitions (workers)
def planner_node(state: AgentState):
    print(f"This is the Planner, now planning research for '{state['topic']}'.") # checkbox

    # Use Tool #2 here!
    today = get_current_date()

    prompt = (
            f"Today is {today}. \n"
            f"Topic: '{state['topic']}'\n"
            f"Break this into 3 distinct, actionable web search queries."
            f"Return ONLY the queries as a bulleted list.")
    
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

            # To Check and Fix the output format from Google Gemini
            # Sometimes Google returns a string, sometimes a list with a signature.
            # This checks which one it is and extract only the text.
            if isinstance(results, list):
                content = results[0].get('content', str(results))
            else:
                content = str(results)
                
            research_results.append(f"Source ({step}): {content}")

        except Exception as e:
            research_results.append(f"Error searching for {step}: {e}")

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
# Routing logic based on data quality
def router_logic(state: AgentState) -> Literal["responder", "planner"]:
    """
    Decides based on data quality:
    - If data exists -> Go to Responder
    - If NO data (and haven't retried too much) -> Go back to Planner
    """

    data_count = len(state.get("research_data", []))
    loop_count = state.get("loop_count", 0)
    
    if data_count > 0 or loop_count >= 2:
        return "responder"
    else:
        print(f" Decision: No data found. Now Retrying... (Attempt {loop_count}) ---")
        return "planner"


# Step 5
# Now, set up the StateGraph to connect everything together.
workflow = StateGraph(AgentState)

# Add the workers to the graph
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("responder", responder_node)

# Flow definition - the order in which nodes are executed
workflow.set_entry_point("planner")        # Start with the Planner
workflow.add_edge("planner", "researcher") # Then go to Researcher

# Logic to decide next step based on research results
workflow.add_conditional_edges(
    "researcher",
    router_logic,
    {
        "planner": "planner",
        "responder": "responder"
    }
)

workflow.add_edge("responder", END)

# Compile the graph into an executable app
app = workflow.compile()


# Step 6
# Here test the entire setup with a sample topic.
if __name__ == "__main__":
    # Here to change the research topic
    topic = "The current state of Nvidia's AI technology and its future prospects."

    print(f"\n Starting the research based on: {topic}\n")

    # Run the graph
    result = app.invoke({"topic": topic, "messages": [], "loop_count": 0})
    
    print("\n" + "="*40)
    print(" Here is the final answer:\n")
    print("="*40)
    print(result["final_answer"])