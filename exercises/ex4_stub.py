"""
Exercise 4 — Parallel map-reduce with Send.

Task:
- State has: regions: list[str], date: str, results: Annotated[list[dict], operator.add]
- spawn_workers conditional edge: returns one Send("process_region", ...) per region
- process_region node: returns {"results": [{"region": x, "rows": <hardcoded>}]}
  Use: {"na": 50000, "eu": 32000, "fe": 18000}
- aggregate node: prints total rows across all regions
- Run with regions=["na", "eu", "fe"], date="2024-01-15"

Fill in the TODOs.
"""
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

class OverallState(TypedDict):
    regions: list[str]
    date: str
    results: Annotated[list[dict], operator.add]

class RegionState(TypedDict):
    region: str
    date: str

# TODO: define spawn_workers(state) → list[Send]

# TODO: define process_region(state: RegionState) → dict

# TODO: define aggregate(state: OverallState) → dict

# TODO: build graph with conditional fan-out and fan-in

# TODO: compile and invoke
