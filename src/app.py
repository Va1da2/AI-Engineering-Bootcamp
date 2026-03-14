import streamlit as st

from core.config import config
from openai import OpenAI
from groq import Groq
from google import genai


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
            max_tokens=max_tokens
        ).choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            reasoning_effort="minimal"
        ).choices[0].message.content

    return response

with st.sidebar:
    st.title("Settings")

    provider = st.selectbox("Provider", ["OpenAI", "Groq", "Google"])
    if provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-5-nano", "gpt-5-mini"])
    elif provider == "Groq":
        model_name = st.selectbox("Model", ["llama-3.3-70b-versatile"])
    else:
        model_name = st.selectbox("Model", ["gemini-2.5-flash"])

    st.session_state.provider = provider
    st.session_state.model_name = model_name

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    response = run_llm(st.session_state.provider, st.session_state.model_name, st.session_state.messages)
    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
