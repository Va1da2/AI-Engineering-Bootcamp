import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient
from langsmith import Client as LSClient

from api.api.models import RAGRequest, RAGResponse, RAGUsedContext, FeedbackRequest, FeedbackResponse
from api.api.processors.submit_feedback import submit_feedback
from api.agents.graph import stream_agent_wrapper


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url="http://qdrant:6333")
langsmith_client = LSClient()

agent_router = APIRouter()
feedback_router = APIRouter()

@agent_router.post("/")
def agent(request: Request, payload: RAGRequest) -> StreamingResponse:

    return StreamingResponse(
        stream_agent_wrapper(payload.query, payload.thread_id, qdrant_client),
        media_type="text/event-stream"
    )

@feedback_router.post("/")
def send_feedback(request: Request, payload: FeedbackRequest) -> FeedbackResponse:
    try:
        submit_feedback(
            payload.trace_id, 
            langsmith_client,
            payload.feedback_score,
            payload.feedback_text,
            payload.feedback_source_type)
    except Exception as e:
        logger.error(f"Encounterred issues with sending feedback: {e}")
        return FeedbackResponse(message="Couldn't send feedback this time. Try later.")

    return FeedbackResponse(message="Success")


api_router = APIRouter()
api_router.include_router(agent_router, prefix="/agent", tags=["agent"])
api_router.include_router(feedback_router, prefix="/submit-feedback", tags=["feedback"])