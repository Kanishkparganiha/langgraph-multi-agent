"""
Exercise 2 — StateGraph with conditional routing.

Task:
- State: {"date": str, "row_count": int, "log": list[str]}
  - log should use operator.add reducer
- Nodes:
    count_rows   → sets row_count = 75000, appends to log
    full_load    → prints "running full load", appends to log
    sample_load  → prints "running sample load", appends to log
- Routing: if row_count > 50000 → full_load, else → sample_load
- Both load nodes go to END
- Print final log at the end

Fill in the TODOs.
"""
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    date: str
    row_count: int
    log: Annotated[list[str], operator.add]

# TODO: define count_rows node

# TODO: define full_load node

# TODO: define sample_load node

# TODO: define routing function

# TODO: build StateGraph, add nodes, add edges, add conditional edge

# TODO: compile and invoke with {"date": "2024-01-15", "row_count": 0, "log": []}

# TODO: print result["log"]
