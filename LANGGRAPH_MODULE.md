# LangChain + LangGraph — Multi-Agent & Memory Module

You've built agentic systems (RISC Strands, F1 tracker, DE Copilot) so you understand the perceive→reason→act loop.
LangGraph is just a different way to express the same thing — as an explicit **state machine graph** instead of an
imperative loop. That's the mental model to carry in.

**Docs reference:** https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/

---

## Setup

```bash
cd /Users/kanishk/Downloads/WorkingDir/langgraph-practice
python3 -m venv venv && source venv/bin/activate
pip install langgraph langchain langchain-community langchain-ollama
```

Uses local Ollama (llama3.2:3b) — no API key needed.

---

## Part 1 — LangGraph Core Concepts

### The mental model

```
Your RISC Strands agent:    LangGraph equivalent:
─────────────────────────   ──────────────────────────────────
@tool decorated functions   → nodes
Strands Agent() loop        → StateGraph with edges
Agent state dict            → TypedDict state schema
Session context             → checkpointer (thread_id)
```

### State — everything passes through it

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# Option A — custom state
class PipelineState(TypedDict):
    asin_list: list[str]
    processed_count: int
    status: str

# Option B — MessagesState (for conversational agents, most common)
from langgraph.graph import MessagesState
# Gives you: messages: Annotated[list[BaseMessage], add_messages]
# add_messages is a reducer — appends new messages instead of replacing

class AgentState(MessagesState):
    # Add your own fields on top
    current_agent: str
    context: dict
```

**Reducers** — the non-obvious part:
```python
# Without reducer: new value REPLACES old value
class State(TypedDict):
    count: int   # state["count"] = 5 replaces it

# With reducer: function decides how to merge
from typing import Annotated
import operator

class State(TypedDict):
    count: Annotated[int, operator.add]   # state["count"] += 5 (accumulates)
    messages: Annotated[list, add_messages]  # appends, deduplicates by id
```

---

### StateGraph — the graph itself

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)

# Nodes — any callable that takes state and returns partial state update
def my_node(state: AgentState) -> dict:
    return {"current_agent": "worker_a"}   # only update what changed

builder.add_node("worker_a", my_node)
builder.add_node("worker_b", another_node)

# Edges — static routing
builder.add_edge(START, "worker_a")
builder.add_edge("worker_a", "worker_b")
builder.add_edge("worker_b", END)

# Conditional edges — dynamic routing
def router(state: AgentState) -> str:
    if state["status"] == "done":
        return END
    return "worker_b"

builder.add_conditional_edges("worker_a", router)

graph = builder.compile()
result = graph.invoke({"messages": [], "current_agent": "", "context": {}})
```

---

## Part 2 — Memory (the interview topic)

LangGraph has **two distinct memory systems**. Interviewers ask this and most people blur them.

### Short-term memory — Checkpointer

Stores the **graph state at every step** for a given `thread_id`. Lets you:
- Resume a conversation after interruption
- Time-travel back to a previous state
- Run graph.get_state() to inspect what happened

```python
from langgraph.checkpoint.memory import InMemorySaver  # dev only
# from langgraph.checkpoint.postgres import PostgresSaver  # prod

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# thread_id scopes the conversation — same thread = same memory
config = {"configurable": {"thread_id": "user-kanishk-session-1"}}

# Turn 1
graph.invoke({"messages": [{"role": "user", "content": "I need to process ASINs"}]}, config)

# Turn 2 — graph automatically has all prior messages in state
graph.invoke({"messages": [{"role": "user", "content": "filter to electronics only"}]}, config)

# Inspect state
snapshot = graph.get_state(config)
print(snapshot.values["messages"])   # full conversation history

# Time-travel — go back 2 steps
history = list(graph.get_state_history(config))
old_config = history[2].config
graph.invoke(None, old_config)   # resume from that checkpoint
```

**When to use:** Multi-turn conversation context, resumable workflows, audit trail.
**Backend:** InMemorySaver (dev), PostgresSaver / RedisSaver (prod).

---

### Long-term memory — Store

Persists facts **across sessions/threads**. The checkpointer forgets when the thread ends.
The store remembers forever (or until you delete).

```python
from langgraph.store.memory import InMemoryStore
import uuid

store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)

# In a node — access via config
def agent_node(state: AgentState, config: dict) -> dict:
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "preferences")    # hierarchical key

    # Read memories
    memories = store.search(namespace, query="ASIN processing preferences")

    # Write a memory
    store.put(namespace, str(uuid.uuid4()), {
        "fact": "User always wants electronics filtered first",
        "created_at": "2024-01-15"
    })
    return {}
```

**Namespace pattern:** `(user_id, category)` → like S3 prefix structure.
**When to use:** User preferences, learned facts, cross-session context.

### The interview answer on memory

> "LangGraph separates memory into two layers. The checkpointer is like a WAL — it snapshots graph state
> at every step for a thread_id, enabling resumable workflows and time-travel debugging. The store is
> long-term memory that persists across threads — user preferences, learned facts. In prod you'd back the
> checkpointer with Postgres and the store with a vector DB for semantic search. Same separation I used in
> RISC — session context vs the agent's accumulated knowledge about a product."

---

## Part 3 — Multi-Agent Patterns

### Pattern 1 — Supervisor (what you explain in interviews)

One supervisor LLM decides which worker agent to call next. Workers report back to supervisor.
Most common pattern for interview questions.

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import AIMessage
from typing import Literal

# Each worker is a ReAct agent with its own tools
def make_worker(llm, tools, system_prompt):
    return create_react_agent(model=llm, tools=tools, prompt=system_prompt)

ingestion_agent = make_worker(llm, [read_s3, write_redshift], "You ingest data from S3 to Redshift.")
quality_agent   = make_worker(llm, [run_dq_check, alert_slack], "You run data quality checks.")

# Worker nodes — run agent, then send back to supervisor
def ingestion_node(state: MessagesState) -> Command:
    result = ingestion_agent.invoke(state)
    return Command(
        update={"messages": [AIMessage(content=result["messages"][-1].content, name="ingestion")]},
        goto="supervisor",
    )

def quality_node(state: MessagesState) -> Command:
    result = quality_agent.invoke(state)
    return Command(
        update={"messages": [AIMessage(content=result["messages"][-1].content, name="quality")]},
        goto="supervisor",
    )

# Supervisor — routes to worker or finishes
from typing import Annotated
class Router(TypedDict):
    next: Literal["ingestion", "quality", "FINISH"]
    reasoning: str

def supervisor_node(state: MessagesState) -> Command:
    system = """You are a DE pipeline supervisor. Route to:
    - ingestion: to load data from S3
    - quality: to validate data quality
    - FINISH: when the pipeline is complete"""

    response = llm.with_structured_output(Router).invoke(
        [{"role": "system", "content": system}] + state["messages"]
    )
    goto = END if response["next"] == "FINISH" else response["next"]
    return Command(goto=goto, update={"messages": []})

# Wire it up
builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("ingestion", ingestion_node)
builder.add_node("quality", quality_node)

graph = builder.compile(checkpointer=InMemorySaver())
```

---

### Pattern 2 — Handoff (peer-to-peer)

Agents hand off directly to each other without a central supervisor. More flexible, harder to debug.

```python
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedToolCallId
from typing import Annotated

# Handoff tool — an agent calls this to transfer control
@tool
def transfer_to_quality_agent(
    summary: Annotated[str, "What was ingested and why quality check is needed"],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer to the data quality agent."""
    return Command(
        goto="quality_agent",
        update={
            "messages": [ToolMessage(content=f"Transferred. Context: {summary}", tool_call_id=tool_call_id)],
            "active_agent": "quality_agent",
        },
        graph=Command.PARENT,   # signal to parent graph, not just subgraph
    )

# Ingestion agent now has this as one of its tools
ingestion_agent = create_react_agent(
    model=llm,
    tools=[read_s3, write_redshift, transfer_to_quality_agent],
    prompt="Ingest data. When done, transfer to quality agent."
)
```

---

### Pattern 3 — Parallel subgraphs (map-reduce style)

```python
from langgraph.constants import Send

class OverallState(TypedDict):
    regions: list[str]
    results: Annotated[list, operator.add]   # accumulates across parallel branches

def spawn_regional_workers(state: OverallState) -> list[Send]:
    # Send = dynamic edge — one per region, all run in parallel
    return [Send("process_region", {"region": r}) for r in state["regions"]]

def process_region(state: dict) -> dict:
    region = state["region"]
    # ... process
    return {"results": [{"region": region, "status": "ok"}]}

builder = StateGraph(OverallState)
builder.add_node("process_region", process_region)
builder.add_conditional_edges(START, spawn_regional_workers)  # fan-out
builder.add_edge("process_region", END)                       # fan-in via reducer

# Same as your A2H pattern: 15 Glue jobs × 3 profiles × 5 regions in parallel
```

---

## Part 4 — create_react_agent (the shortcut)

For a single agent with tools, you don't need to build the graph manually:

```python
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

llm = ChatOllama(model="llama3.2:3b")

@tool
def get_row_count(table_name: str) -> int:
    """Get row count for a table."""
    counts = {"site_metrics": 300, "labor_events": 300}
    return counts.get(table_name, 0)

@tool
def check_freshness(table_name: str) -> str:
    """Check when a table was last updated."""
    return f"{table_name} was updated 2 hours ago"

agent = create_react_agent(
    model=llm,
    tools=[get_row_count, check_freshness],
    prompt="You are a data quality assistant. Use tools to check table health.",
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "session-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "check site_metrics health"}]},
    config
)
print(result["messages"][-1].content)
```

---

## Part 5 — Interview Framing

### "Have you used LangChain/LangGraph?"

> "I haven't used LangGraph in production — our team uses AWS Strands with Bedrock. But the patterns are
> identical: Strands uses @tool decorators and an Agent() loop; LangGraph externalises that loop as an
> explicit StateGraph. I've built the RISC agentic pipeline with Strands — supervisor routing, tool calling,
> IAM role chaining for cross-account access — and I've experimented with LangGraph locally to understand
> the differences. The main thing LangGraph adds is the checkpointer model for resumable workflows and
> time-travel debugging, which Strands doesn't have natively."

### Checkpointer vs Store

> "Checkpointer is session memory — every step's state is snapshotted for a thread_id, like a transaction
> log. Store is long-term memory — facts that survive across sessions, backed by a key-value or vector store.
> In production you'd checkpoint to Postgres and store to Pinecone or pgvector for semantic retrieval."

### Supervisor vs Handoff

> "Supervisor is centralised control — one LLM sees everything and decides routing. Easier to debug,
> single point of failure. Handoff is peer-to-peer — agents decide themselves when to transfer. More
> flexible but harder to trace. For DE workflows I'd use supervisor because auditability matters more
> than flexibility."

---

## Exercises

### Exercise 1 — Build a single ReAct agent

Write a `create_react_agent` that has two tools:
- `list_tables()` → returns `["site_metrics", "labor_events", "pipeline_runs"]`
- `get_row_count(table_name: str)` → returns a hardcoded dict lookup

Enable memory (InMemorySaver). Run two turns:
1. "what tables do we have?"
2. "how many rows does labor_events have?"

Verify the agent remembers turn 1 in turn 2 (it shouldn't call list_tables again).

→ Solution: `exercises/ex1_solution.py`

---

### Exercise 2 — StateGraph with conditional routing

Build a StateGraph with:
- State: `{"query": str, "row_count": int, "action": str}`
- Node `count_rows`: sets `row_count = 75000` (hardcoded)
- Node `full_load`: prints "running full load"
- Node `sample_load`: prints "running sample load"
- Conditional edge from `count_rows`: if row_count > 50000 → full_load, else → sample_load
- Both load nodes go to END

→ Solution: `exercises/ex2_solution.py`

---

### Exercise 3 — Two-agent supervisor

Build a supervisor graph with:
- `ingestion_agent` — has tool `read_from_s3(date: str) -> str` (returns "read 50000 rows")
- `quality_agent` — has tool `run_dq(table: str) -> str` (returns "0 nulls found")
- `supervisor_node` — routes to ingestion, quality, or FINISH
- Memory enabled — conversation persists across invocations

Test with: "ingest data for 2024-01-15 then check quality"

→ Solution: `exercises/ex3_solution.py`

---

### Exercise 4 — Parallel map-reduce

Build a graph that:
- Starts with `regions = ["na", "eu", "fe"]` in state
- Fans out to process each region in parallel using `Send`
- Each regional node returns `{"results": [{"region": x, "rows": 1000}]}`
- Results accumulate via reducer
- Final node prints total rows across all regions

→ Solution: `exercises/ex4_solution.py`

---

## Interview Questions — Answer First

1. What's the difference between `add_messages` reducer and a plain list in state? When do you need a reducer?

2. You have a 10-turn conversation. The user asks "what did I say at the start?" How does LangGraph know?

3. Supervisor vs handoff — which would you use for a DE pipeline that needs audit logs? Why?

4. Your agent graph runs 50 nodes for a batch job. Halfway through it crashes. How do you resume from step 25?

5. You want the agent to remember a user's table preferences across different days/sessions. Checkpointer or store?

6. What's `Command(graph=Command.PARENT)` for? When would you need it?

7. In the supervisor pattern, why do worker nodes return `Command(goto="supervisor")` instead of just returning state?
