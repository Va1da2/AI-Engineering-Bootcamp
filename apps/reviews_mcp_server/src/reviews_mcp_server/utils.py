import openai

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch,
    Filter,
    MatchAny,
    FieldCondition,
    FusionQuery
)


def get_embedding(text, model='text-embedding-3-small'):

    response = openai.embeddings.create(
        input=text,
        model=model
    )

    return response.data[0].embedding

def retrieve_prefiltered_review_data(query: str, parent_asins: list[str], k=5) -> dict:

    qdrant_client = QdrantClient(url="http://qdrant:6333")
   
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-reviews-collection-01",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(
                                any=parent_asins
                            )
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion='rrf'),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["preprocessed_data"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores
    }

def process_reviews(reviews: dict) -> str:
    formatted_reviews = ""
    for id, review in zip(reviews["retrieved_context_ids"], reviews["retrieved_context"]):
        formatted_reviews += f"- ID: {id}, review: {review}\n"
    
    return formatted_reviews