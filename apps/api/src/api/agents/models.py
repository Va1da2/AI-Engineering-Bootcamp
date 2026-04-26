from typing import Any, Annotated, List
from operator import add

from pydantic import BaseModel, Field


class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class FinalResponse(BaseModel):
    answer: str = Field(description="Answer the the question")
    references: List[RAGUsedContext] = Field(description="List of items used to answer the question")

class IntentRouterNode(BaseModel):
    question_relevant: bool
    answer: str

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    final_answer: bool = False
    references: Annotated[List[RAGUsedContext], add] = []
    trace_id: str = ""