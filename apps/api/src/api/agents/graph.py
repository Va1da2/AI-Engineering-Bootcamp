import json
import numpy as np

from qdrant_client.models import Filter, FieldCondition, MatchValue

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver

from api.agents.agents import agent_node, intent_router_node
from api.agents.tools import get_formatted_item_context, get_formatted_item_reviews
from api.agents.models import State
from api.agents.utils.utils import process_graph_event, string_for_sse


def intent_router_conditional_edges(state: State) -> str:

    if state.question_relevant:
        return "agent_node"
    
    return "end"

def tool_router(state: State) -> str:
    
    if state.final_answer:
        return "end"
    
    if state.iteration > 2:
        return "end"
    
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    
    return "end"


workflow = StateGraph(State)
workflow.add_node("tool_node", ToolNode([get_formatted_item_context, get_formatted_item_reviews]))
workflow.add_node("agent_node", agent_node)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")
workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END
    }
)
workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "end": END
    }
)
workflow.add_edge("tool_node", "agent_node")


def stream_agent_wrapper(question: str, thread_id: str, qdrant_client, top_k=5):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "iteration": 0
    }

    with PostgresSaver.from_conn_string(
        "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"
    ) as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
       
        for chunk in graph.stream(initial_state, config=config, stream_mode=["debug", "values"]):
            processed_chunk = process_graph_event(chunk)

            if processed_chunk:
                yield string_for_sse(processed_chunk)

            if chunk[0] == "values":
                result = chunk[1]

    used_context = []
    dummy_vector = np.zeros(1536)

    for item in result.get("references", []):
        payload = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            query=dummy_vector,
            using="text-embedding-3-small",
            limit=1,
            with_payload=True,
            with_vectors=False,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.get("id"))
                    )
                ]
            )
        ).points[0].payload

        image_url = payload.get("image")
        price = payload.get("price")
        if image_url:
            used_context.append({
                "image_url": image_url,
                "price": price,
                "description": item.get("description")
            })

    yield string_for_sse(
        json.dumps(
            {
                "type": "final_answer",
                "data": {
                    "answer": result.get("answer"),
                    "used_context": used_context,
                    "trace_id": result.get("trace_id", "")
                }
            }
        )
    )