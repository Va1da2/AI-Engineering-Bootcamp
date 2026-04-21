import instructor

from typing import List

from jinja2 import Template
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable, get_current_run_tree

from api.agents.tools import get_formatted_item_context
from api.agents.utils.prompt_management import from_template_config


class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class FinalResponse(BaseModel):
    answer: str = Field(description="Answer the the question")
    references: List[RAGUsedContext] = Field(description="List of items used to answer the question")

class IntentRouterNode(BaseModel):
    question_relevant: bool
    answer: str


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def agent_node(state) -> dict:
    
    prompt = from_template_config("api/agents/prompts/shopping_assistant.yaml", "shopping_assistant").render()

    llm = ChatOpenAI(model="gpt-4.1-mini").bind_tools(
        [get_formatted_item_context, FinalResponse],
        tool_choice="auto"
    )

    response = llm.invoke(
        [
            SystemMessage(content=prompt),
            *state.messages
        ]
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.response_metadata["token_usage"]["prompt_tokens"],
            "total_tokens": response.response_metadata["token_usage"]["total_tokens"],
            "completion_tokens": response.response_metadata["token_usage"]["completion_tokens"]
        }

    final_answer = False
    answer = ""
    references = []
    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalResponse":
                final_answer = True
                answer = tool_call.get("args").get("answer")
                references.extend(tool_call.get("args").get("references"))
    
    return {
        "messages": [response],
        "iteration": state.iteration + 1,
        "answer": answer,
        "final_answer": final_answer,
        "references": references
    }

@traceable(
    name="intent_router",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def intent_router_node(state) -> dict:

    prompt_template = from_template_config("api/agents/prompts/intent_router.yaml", "intent_router")

    prompt = prompt_template.render(
        question=state.messages[0].content
        )

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS
    )

    response, raw_response = client.create_with_completion(
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        response_model=IntentRouterNode
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer
    }