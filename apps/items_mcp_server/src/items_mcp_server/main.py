from fastmcp import FastMCP

from items_mcp_server.utils import retrieve_items_data, process_context


mcp = FastMCP("items_mcp_server")

@mcp.tool()
def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """Get the context for top k items - each item is an inventory item for a given query.

    Args:
        query: The query to get the top k items for
        top_k: The number of items and context to retrieve, works best with 5 or more

    Returns:
        A string representing context for top_k items from inventory for a given query. Information returned - IDs, average rating and description of item.
    """
    context = retrieve_items_data(query, top_k)

    return process_context(context)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)