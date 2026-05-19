"""
Exercise 3 — Two-agent supervisor.

Task:
- ingestion_agent with tool: read_from_s3(date: str) → "read 50000 rows"
- quality_agent with tool: run_dq(table: str) → "0 nulls found"
- supervisor_node routes to "ingestion", "quality", or FINISH using structured output
- Memory enabled (InMemorySaver)
- Test: "ingest data for 2024-01-15 then check quality on site_metrics"
- Print each supervisor routing decision

Hint: use Command(goto=...) in worker nodes to return to supervisor.
Fill in the TODOs.
"""
from typing import Literal
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

llm = ChatOllama(model="llama3.2:3b")

# TODO: define read_from_s3 tool

# TODO: define run_dq tool

# TODO: create ingestion_agent

# TODO: create quality_agent

# TODO: define ingestion_node (runs ingestion_agent, returns Command goto="supervisor")

# TODO: define quality_node (runs quality_agent, returns Command goto="supervisor")

# TODO: define Router TypedDict with next: Literal["ingestion", "quality", "FINISH"]

# TODO: define supervisor_node (uses llm.with_structured_output(Router), returns Command)

# TODO: build graph: START → supervisor → [ingestion, quality] → supervisor → END

graph = None  # replace with compiled graph

config = {"configurable": {"thread_id": "ex3-run"}}

if graph:
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "ingest data for 2024-01-15 then check quality on site_metrics"}]},
        config,
    )
    print(result["messages"][-1].content)
