from pydantic import BaseModel


class ChatRequest(BaseModel):
    provider: str
    model_name: str
    messages: list[dict]

class ChatResponse(BaseModel):
    message: str
