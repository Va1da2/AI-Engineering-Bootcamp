import logging

from fastapi import APIRouter, Request
from qdrant_client import QdrantClient

from api.api.models import RAGRequest, RAGResponse, RAGUsedContext
from api.agents.graph import run_agent_wrapper


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url="http://qdrant:6333")

agent_router = APIRouter()

@agent_router.post("/")
def agent(request: Request, payload: RAGRequest) -> RAGResponse:

    result = run_agent_wrapper(payload.query, payload.thread_id, qdrant_client)

    return RAGResponse(
        answer=result["answer"],
        used_context=[RAGUsedContext(**item) for item in result["used_context"]]
    )


api_router = APIRouter()
api_router.include_router(agent_router, prefix="/agent", tags=["agent"])