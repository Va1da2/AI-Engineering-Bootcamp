import openai

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Document, RrfQuery, Rrf
from langchain.tools import tool
from langsmith import traceable, get_current_run_tree


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model": "text_embedding_3_small"}
)
def get_embedding(text, model='text-embedding-3-small'):

    response = openai.embeddings.create(
        input=text,
        model=model
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }

    return response.data[0].embedding

@traceable(
    name="retrieve_data",
    run_type='retriever'
)
def retrieve_data(query, k=5):

    qdrant_client = QdrantClient(url="http://qdrant:6333")
   
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25",
                ),
                using="bm25",
                limit=20
            )
        ],
        query=RrfQuery(rrf=Rrf(weights=[1,1])),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []
    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload["average_rating"])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }

@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):
    
    formatted_context = ""

    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context

@tool
def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """Get the context for top k items - each item is an inventory item for a given query.

    Args:
        query: The query to get the top k items for
        top_k: The number of items and context to retrieve, works best with 5 or more

    Returns:
        A string representing context for top_k items from inventory for a given query. Information returned - IDs, average rating and description of item.
    """
    context = retrieve_data(query, top_k)

    return process_context(context)