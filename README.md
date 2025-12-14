# LangGraph Research Assistant Agent

## Overview
This is an autonomous AI agent built with LangGraph that plans, researches, and writes reports on any given topic. It features a multi-step workflow with conditional retries and human-in-the-loop validation.

## Architecture
- **Planner Node:** Decomposes the topic into search queries.
- **Researcher Node:** Executes searches using DuckDuckGo.
- **Router Logic:** Checks if data was found. If not, it loops back to the Planner (max 2 retries).
- **Responder Node:** Summarizes findings into a report.
- **Human Loop:** Pauses before writing the final report to ask for user confirmation.

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a .env file and add your Google API key:
    ```bash
    GOOGLE_API_KEY=your_key_here
    ```
4. Run the Agent:
    ```bash
    python main.py
    ```
