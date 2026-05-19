"""
Example 4 — Supervisor multi-agent pattern.
Shows: Command, structured routing, worker agents, supervisor node.
Run: python examples/04_supervisor.py
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

# ── Worker tools ──────────────────────────────────────────────────────────────

@tool
def read_from_s3(date: str) -> str:
    """Read raw data from S3 for a given date."""
    return f"Read 50,000 rows from s3://bucket/input/{date}/data.parquet"

@tool
def write_to_redshift(table: str, row_count: int) -> str:
    """Write processed data to Redshift."""
    return f"Wrote {row_count} rows to {table} successfully"

@tool
def run_dq_check(table: str) -> str:
    """Run data quality checks on a Redshift table."""
    return f"DQ check on {table}: 0 nulls, 0 duplicates, freshness OK"

@tool
def alert_slack(message: str) -> str:
    """Send an alert to the Slack channel."""
    return f"Alert sent: {message}"

# ── Worker agents ─────────────────────────────────────────────────────────────

ingestion_agent = create_react_agent(
    model=llm,
    tools=[read_from_s3, write_to_redshift],
    prompt="You are a data ingestion agent. Read from S3 and write to Redshift.",
)

quality_agent = create_react_agent(
    model=llm,
    tools=[run_dq_check, alert_slack],
    prompt="You are a data quality agent. Run DQ checks and alert on failures.",
)

# ── Worker nodes ──────────────────────────────────────────────────────────────

def ingestion_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = ingestion_agent.invoke(state)
    return Command(
        update={"messages": [AIMessage(content=result["messages"][-1].content, name="ingestion")]},
        goto="supervisor",
    )

def quality_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = quality_agent.invoke(state)
    return Command(
        update={"messages": [AIMessage(content=result["messages"][-1].content, name="quality")]},
        goto="supervisor",
    )

# ── Supervisor ────────────────────────────────────────────────────────────────

class Router(TypedDict):
    next: Annotated[Literal["ingestion", "quality", "FINISH"], "which worker to call next"]
    reasoning: str

def supervisor_node(state: MessagesState) -> Command[Literal["ingestion", "quality", "__end__"]]:
    system = """You are a DE pipeline supervisor. Route to:
- ingestion: when data needs to be loaded from S3 to Redshift
- quality: when data quality checks need to run
- FINISH: when the pipeline is fully complete

Look at the conversation history to decide what's already been done."""

    response = llm.with_structured_output(Router).invoke(
        [{"role": "system", "content": system}] + state["messages"]
    )
    goto = END if response["next"] == "FINISH" else response["next"]
    print(f"[supervisor] → {response['next']} | reason: {response['reasoning']}")
    return Command(goto=goto)

# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("ingestion", ingestion_node)
builder.add_node("quality", quality_node)

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "pipeline-run-1"}}

print("=== Running DE pipeline via supervisor ===\n")
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Ingest data for 2024-01-15 then run quality checks"}]},
    config,
)
print("\n=== Final answer ===")
print(result["messages"][-1].content)
