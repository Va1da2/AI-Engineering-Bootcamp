from openai import OpenAI
from groq import Groq
from google import genai

from api.core.config import config


def run_llm(provider: str, model_name: str, messages: list[dict], max_tokens: int=500) -> str:

    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    elif provider == "Groq":
        client = Groq(api_key=config.GROQ_API_KEY)
    else:
        client = genai.Client(api_key=config.GOOGLE_API_KEY)

    if provider == "Google":
        response = client.models.generate_content(
            model=model_name,
            contents=[message["content"] for message in messages]
        ).text
    elif provider == "Groq":
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens
        ).choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning_effort="minimal"
        ).choices[0].message.content

    return response
