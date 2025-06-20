from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# Define the state for the supervisor graph
class SupervisorState(TypedDict):
    user_query: str
    csv_file_content: str # Pass through csv content
    route: Literal["prediction", "conversation"]

# Define a node to make the routing decision
def route_node(state: SupervisorState) -> SupervisorState:
    """Decides the route based on the user's query."""
    query = state.get("user_query", "").lower()
    print(f"[Supervisor] Routing query: '{query}'")
    if "예측" in query or "prediction" in query or "predict" in query:
        print("[Supervisor] Decision: route to prediction")
        return {**state, "route": "prediction"}
    else:
        # For now, we'll just have a placeholder for a general conversational agent
        print("[Supervisor] Decision: route to conversation")
        return {**state, "route": "conversation"}

# Define the graph
builder = StateGraph(SupervisorState)
builder.add_node("router", route_node)
builder.set_entry_point("router")
builder.add_edge("router", END)

# Compile the graph
supervisor_app = builder.compile()
