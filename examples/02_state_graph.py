"""
Example 2 — StateGraph with conditional routing.
Shows: nodes, edges, conditional_edges, state reducers.
Run: python examples/02_state_graph.py
"""
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

class PipelineState(TypedDict):
    date: str
    row_count: int
    results: Annotated[list[str], operator.add]   # reducer — accumulates across nodes

def count_rows(state: PipelineState) -> dict:
    count = 75_000   # in real life: query Redshift
    print(f"[count_rows] found {count} rows for {state['date']}")
    return {"row_count": count, "results": [f"counted {count} rows"]}

def full_load(state: PipelineState) -> dict:
    print(f"[full_load] processing all {state['row_count']} rows")
    return {"results": ["ran full load"]}

def sample_load(state: PipelineState) -> dict:
    sample = state["row_count"] // 10
    print(f"[sample_load] sampling {sample} rows")
    return {"results": [f"ran sample load ({sample} rows)"]}

def route_by_size(state: PipelineState) -> str:
    """Conditional edge function — returns the name of the next node."""
    if state["row_count"] > 50_000:
        return "full_load"
    return "sample_load"

builder = StateGraph(PipelineState)
builder.add_node("count_rows",  count_rows)
builder.add_node("full_load",   full_load)
builder.add_node("sample_load", sample_load)

builder.add_edge(START, "count_rows")
builder.add_conditional_edges("count_rows", route_by_size)   # dynamic routing
builder.add_edge("full_load",   END)
builder.add_edge("sample_load", END)

graph = builder.compile()

result = graph.invoke({"date": "2024-01-15", "row_count": 0, "results": []})
print(f"\nFinal results: {result['results']}")
