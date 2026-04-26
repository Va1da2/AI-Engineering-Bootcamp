from fastmcp import FastMCP
from qdrant_client import QdrantClient

from reviews_mcp_server.utils import retrieve_prefiltered_review_data, process_reviews



mcp = FastMCP("reviews_mcp_server")

@mcp.tool()
def get_formatted_item_reviews(query: str, items: list[str], top_k: int = 5) -> str:
    """Get the top k reviews matching a query for a list of prefiltered items.

    Args:
        query: The query to get the top k reviews for
        items: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retreieve, this should be at least 20 if multiple items are prefiltered
    
    Returns:
        A string of the top k reviews with IDs prepending each review. Each line is a single review for one of the items in the items list.
    """

    reviews = retrieve_prefiltered_review_data(query, items, top_k)

    return process_reviews(reviews)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)