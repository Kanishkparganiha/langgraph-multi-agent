"""
Example 3 — Short-term vs long-term memory.
Shows: checkpointer (session), store (cross-session), time-travel.
Run: python examples/03_memory.py
"""
import uuid
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage

llm = ChatOllama(model="llama3.2:3b")

# Short-term: scoped to thread_id, lost when thread ends
checkpointer = InMemorySaver()

# Long-term: persists across threads / sessions
store = InMemoryStore()

def agent_node(state: MessagesState, config: dict) -> dict:
    user_id = config["configurable"].get("user_id", "default")
    namespace = (user_id, "preferences")

    # Read long-term memories
    memories = store.search(namespace, query=state["messages"][-1].content)
    memory_text = "\n".join(m.value["fact"] for m in memories) if memories else "No memories yet."

    system = f"You are a helpful DE assistant.\nUser memories:\n{memory_text}"
    response = llm.invoke([{"role": "system", "content": system}] + state["messages"])

    # Save something to long-term memory
    store.put(namespace, str(uuid.uuid4()), {
        "fact": f"User asked: {state['messages'][-1].content}"
    })

    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

graph = builder.compile(checkpointer=checkpointer, store=store)

config = {"configurable": {"thread_id": "session-1", "user_id": "kanishk"}}

print("=== Session 1 — Turn 1 ===")
r = graph.invoke({"messages": [HumanMessage(content="I prefer processing NA region first")]}, config)
print(r["messages"][-1].content)

print("\n=== Session 1 — Turn 2 (same thread — checkpointer has history) ===")
r = graph.invoke({"messages": [HumanMessage(content="what did I just say?")]}, config)
print(r["messages"][-1].content)

# New thread — checkpointer forgets, but store remembers
config2 = {"configurable": {"thread_id": "session-2", "user_id": "kanishk"}}
print("\n=== Session 2 — New thread, but store has long-term memories ===")
r = graph.invoke({"messages": [HumanMessage(content="what do you know about my preferences?")]}, config2)
print(r["messages"][-1].content)

# Time-travel
print("\n=== Time-travel — state history for session-1 ===")
history = list(graph.get_state_history(config))
print(f"Steps recorded: {len(history)}")
for snap in history:
    msg_count = len(snap.values.get("messages", []))
    print(f"  step {snap.metadata.get('step')}: {msg_count} messages")
