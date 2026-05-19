"""
Example 1 — Single ReAct agent with tools and memory.
Run: python examples/01_simple_agent.py
"""
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

llm = ChatOllama(model="llama3.2:3b")

@tool
def list_tables() -> list[str]:
    """List all available data tables."""
    return ["site_metrics", "labor_events", "pipeline_runs"]

@tool
def get_row_count(table_name: str) -> int:
    """Get the number of rows in a table."""
    counts = {"site_metrics": 300, "labor_events": 300, "pipeline_runs": 50}
    return counts.get(table_name, -1)

@tool
def check_freshness(table_name: str) -> str:
    """Check when a table was last updated."""
    freshness = {
        "site_metrics": "2 hours ago",
        "labor_events": "27 hours ago — STALE",
        "pipeline_runs": "1 hour ago",
    }
    return freshness.get(table_name, "unknown")

checkpointer = InMemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[list_tables, get_row_count, check_freshness],
    prompt="You are a data quality assistant for a last-mile delivery team. Use tools to answer questions about table health.",
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "demo-session-1"}}

print("=== Turn 1 ===")
r1 = agent.invoke({"messages": [{"role": "user", "content": "what tables do we have?"}]}, config)
print(r1["messages"][-1].content)

print("\n=== Turn 2 (agent should remember tables from turn 1) ===")
r2 = agent.invoke({"messages": [{"role": "user", "content": "which of those tables are stale?"}]}, config)
print(r2["messages"][-1].content)
