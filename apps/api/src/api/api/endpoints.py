import logging

from fastapi import APIRouter, Request
from qdrant_client import QdrantClient

from api.api.models import RAGRequest, RAGResponse
from api.agents.retrieval_generation import RAG_pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url="http://qdrant:6333")

rag_router = APIRouter()

@rag_router.post("/")
def rag(request: Request, payload: RAGRequest) -> RAGResponse:

    result = RAG_pipeline(payload.query, qdrant_client)

    return RAGResponse(answer=result["answer"])


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])