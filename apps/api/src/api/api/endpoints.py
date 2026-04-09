import logging

from fastapi import APIRouter, Request

from api.api.models import ChatRequest, ChatResponse
from api.agents.agents import run_llm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

chat_router = APIRouter()

@chat_router.post("/")
def chat(request: Request, payload: ChatRequest) -> ChatResponse:

    result = run_llm(payload.provider, payload.model_name, payload.messages)

    return ChatResponse(message=result)


api_router = APIRouter()
api_router.include_router(chat_router, prefix="/chat")