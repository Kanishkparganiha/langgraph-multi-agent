"""
Exercise 1 — Single ReAct agent with memory.

Task:
- Create a create_react_agent with two tools:
    list_tables() → ["site_metrics", "labor_events", "pipeline_runs"]
    get_row_count(table_name: str) → hardcoded dict lookup
- Enable InMemorySaver
- Run two turns on the same thread_id:
    Turn 1: "what tables do we have?"
    Turn 2: "how many rows does labor_events have?"
- The agent should NOT call list_tables again in turn 2 (it remembers)

Fill in the TODOs below.
"""
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

llm = ChatOllama(model="llama3.2:3b")

# TODO: define list_tables tool

# TODO: define get_row_count tool

# TODO: create checkpointer

# TODO: create agent with both tools and checkpointer

config = {"configurable": {"thread_id": "ex1-session"}}

# TODO: invoke turn 1 and print result

# TODO: invoke turn 2 and print result
