"""
Research Assistant Agent (Interactive + Robust Version)

Description:
    A fully interactive CLI tool that asks for user input, allows custom instructions,
    and supports feedback loops. It includes automatic retries for API rate limits.
"""

import operator
import os
import sys
import time
from datetime import datetime
from typing import Annotated, List, TypedDict, Literal

# Load environment variables
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 
from fpdf import FPDF 

# Retry Logic Imports
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

load_dotenv()

# Configuration: Use paid stack (Tavily + OpenAI) or free stack (DuckDuckGo + Google Gemini)
use_paid_stack = False 


# 1. Agent State Definition
class AgentState(TypedDict):
    topic: str
    custom_instructions: str
    plan: List[str]
    research_data: List[str]
    final_answer: str
    messages: Annotated[List[BaseMessage], operator.add]
    loop_count: int


# 2. llm and Tool Setup
if use_paid_stack:
    from langchain_openai import ChatOpenAI
    from langchain_community.tools.tavily_search import TavilySearchResults
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    web_search_tool = TavilySearchResults(max_results=2)
    print(" Mode: Paid (OpenAI + Tavily)")

else:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.tools import DuckDuckGoSearchRun
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0)
    
    class DuckDuckGoWrapper:
        def invoke(self, query):
            try:
                search = DuckDuckGoSearchRun()
                result = search.invoke(query)
                return [{"content": result}]
            except Exception as e:
                return [] 
    web_search_tool = DuckDuckGoWrapper()
    print(" Mode: Free (Gemini + DuckDuckGo)")

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")


# 3. Safety wrapper (Prevents 429 Errors: limit retries on rate limit errors)

@retry(
    retry=retry_if_exception_type(ChatGoogleGenerativeAIError),
    wait=wait_fixed(60),  # Wait 60s if rate limit hits
    stop=stop_after_attempt(2),
    before_sleep=lambda retry_state: print(f"\n Rate limit hit. Sleeping for 60s before retry #{retry_state.attempt_number}"))
def call_llm_safely(prompt_messages):
    """Wraps the LLM call with retry logic for rate limits."""
    return llm.invoke(prompt_messages)


# 4. Nodes Definitions

def planner_node(state: AgentState):
    print(f"\n Planner: Thinking about '{state['topic']}'...")
    
    today = get_current_date()
    extra_instructions = state.get("custom_instructions", "")
    
    prompt = (f"Today is {today}. Topic: '{state['topic']}'.\n"
              f"User Instructions: {extra_instructions}\n"
              "Break this into 3 distinct, actionable web search queries. "
              "Return ONLY the queries as a bulleted list.")
    
    # Use the safe wrapper instead of llm.invoke directly
    response = call_llm_safely([HumanMessage(content=prompt)])
    
    content = response.content
    text_content = content[0].get('text', str(content)) if isinstance(content, list) else str(content)
    plan_steps = [line.strip("- *") for line in text_content.split("\n") if line.strip()]
    
    current_loop = state.get("loop_count", 0)
    return {"plan": plan_steps, "loop_count": current_loop + 1}

def researcher_node(state: AgentState):
    print(" Researcher: Searching the web...")
    plan = state.get("plan", [])
    research_results = []
    
    for step in plan:
        print(f"   -> Query: {step}")
        try:
            results = web_search_tool.invoke(step)
            if not results: continue
            content = results[0].get('content', str(results)) if isinstance(results, list) else str(results)
            research_results.append(f"Source ({step}): {content}")
        except Exception:
            continue
    return {"research_data": research_results}

def responder_node(state: AgentState):
    print(" Responder: Writing the report...")
    if not state["research_data"]:
        return {"final_answer": "Error: No research data found."}

    data = "\n".join(state["research_data"])
    extra_instructions = state.get("custom_instructions", "")

    prompt = (
        f"Topic: {state['topic']}\n"
        f"User Specific Instructions: {extra_instructions}\n"
        f"Research Data:\n{data}\n\n"
        f"Write a professional summary based ONLY on the data above. "
        f"Explicitly cite your sources."
    )
    
    # Use the safe wrapper here 
    response = call_llm_safely([HumanMessage(content=prompt)])
    
    content = response.content
    final_text = content[0].get('text', str(content)) if isinstance(content, list) else str(content)
    return {"final_answer": final_text}

def router_logic(state: AgentState) -> Literal["responder", "planner"]:
    if len(state.get("research_data", [])) > 0 or state.get("loop_count", 0) >= 2:
        return "responder"
    print("  No data found. Retrying...")
    return "planner"

def save_to_pdf(text, filename="Research_Report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=text)
    pdf.output(filename)
    print(f"\n📄 PDF saved successfully: {filename}")


# 5. graph and app setup

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("responder", responder_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_conditional_edges("researcher", router_logic, {"planner": "planner", "responder": "responder"})
workflow.add_edge("responder", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["responder"])


# 6. Interactive CLI Loop

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" This is the interactive AI Research Assistant.")
    print("="*50)

    # 1. Ask for Topic
    topic_input = input("\n What topic would you like to research?\n> ")
    if not topic_input: topic_input = "The Future of AI" 

    # 2. Ask for Custom Instructions
    print("\n Any specific instructions? (e.g., 'Focus on financial risks')")
    custom_input = input("> ")
    
    initial_state = {
        "topic": topic_input,
        "custom_instructions": custom_input,
        "messages": [],
        "loop_count": 0
    }
    
    thread_config = {"configurable": {"thread_id": "1"}}
    
    # Full interactive loop with feedback
    while True:
        # Phase A: Plan & Research
        print("\n Starting Agent Workflow...")
        app.invoke(initial_state, config=thread_config)
        
        # Phase B: Review Research
        print("\n" + "-"*20)
        print(" Research is complete (Paused)")
        proceed = input("Now proceed to write the report? (y/n): ")
        
        if proceed.lower() not in ["y", "yes"]:
            print(" Operation cancelled by user.")
            break

        # Phase C: Write Report (Resumes from Pause)
        # Note: This step uses the 'call_llm_safely' wrapper inside, so if
        # it hits a rate limit, it will pause here for 60s automatically.
        result = app.invoke(None, config=thread_config)
        final_answer = result["final_answer"]

        # Phase D: Display & Validate
        print("\n" + "="*20)
        print("Final Draft Report")
        print("="*20)
        print(final_answer)
        print("="*20)

        satisfaction = input("\n Are you satisfied with this result? (y/n): ")
        
        if satisfaction.lower() in ["y", "yes"]:
            save_to_pdf(final_answer)
            print("\n Done! Exiting.")
            break
        else:
            print("\n Okay, let's refine it.")
            feedback = input("What specifically should be changed/added? (This will trigger a re-plan)\n> ")
            
            # Reset for re-planning with feedback
            initial_state["custom_instructions"] = f"{custom_input}. FEEDBACK: {feedback}"
            initial_state["loop_count"] = 0
            
            print("\n Restarting research with new feedback...")