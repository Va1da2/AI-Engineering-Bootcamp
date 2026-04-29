def process_graph_event(chunk):

    def _is_node_start(chunk):
        return chunk[1].get("type") == "task"
    
    def _is_node_end(chunk):
        return chunk[0] == "updates"

    def _tool_to_text(tool_call):
        if tool_call.get("name") == "get_formatted_item_context":
            return f"Looking for items: {tool_call.get("args").get('query', '')}."
        elif tool_call.get("name") == "get_formatted_item_reviews":
            return "Fetching user reviews..."
    
    def _get_chunk_name(chunk):
        return chunk[1].get("payload", {}).get("name")
        
    if _is_node_start(chunk):
        if _get_chunk_name(chunk) == "intent_router_node":
            return "Analysing the question..."
        if _get_chunk_name(chunk) == "agent_node":
            return "Planning..."
        if _get_chunk_name(chunk) == "tool_node":
            message = " ".join([_tool_to_text(tool_call) for tool_call in chunk[1].get("payload", {}).get("input", {}).messages[-1].tool_calls])
            return message

        return ""

def string_for_sse(processed_chunk):
    return f"data: {processed_chunk}\n\n"