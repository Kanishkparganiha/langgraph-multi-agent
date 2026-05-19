"""
Example 5 — Parallel map-reduce with Send (your A2H regional pattern).
Shows: Send, Annotated reducer, fan-out, fan-in.
Run: python examples/05_parallel_mapreduce.py
"""
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

REGIONS = ["na", "eu", "fe", "jp"]

class OverallState(TypedDict):
    regions: list[str]
    date: str
    results: Annotated[list[dict], operator.add]   # reducer — fan-in accumulator

class RegionState(TypedDict):
    region: str
    date: str

def spawn_regional_workers(state: OverallState) -> list[Send]:
    """Conditional edge — returns a Send per region (fan-out)."""
    return [
        Send("process_region", {"region": r, "date": state["date"]})
        for r in state["regions"]
    ]

def process_region(state: RegionState) -> dict:
    """Worker — runs in parallel for each region."""
    rows = {"na": 50_000, "eu": 32_000, "fe": 18_000, "jp": 11_000}[state["region"]]
    print(f"  [process_region] {state['region']} — {rows} rows")
    return {"results": [{"region": state["region"], "rows": rows, "status": "ok"}]}

def aggregate(state: OverallState) -> dict:
    """Fan-in — all regional results are already merged by the reducer."""
    total = sum(r["rows"] for r in state["results"])
    print(f"\n[aggregate] Total rows across {len(state['results'])} regions: {total:,}")
    return {}

builder = StateGraph(OverallState)
builder.add_node("process_region", process_region)
builder.add_node("aggregate", aggregate)

builder.add_conditional_edges(START, spawn_regional_workers)   # fan-out
builder.add_edge("process_region", "aggregate")                # fan-in
builder.add_edge("aggregate", END)

graph = builder.compile()

print("=== Parallel regional processing ===")
result = graph.invoke({
    "regions": REGIONS,
    "date": "2024-01-15",
    "results": [],
})
print(f"\nResults: {result['results']}")
