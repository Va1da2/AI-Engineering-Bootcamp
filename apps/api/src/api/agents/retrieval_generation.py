import openai


def get_embedding(text, model='text-embedding-3-small'):

    response = openai.embeddings.create(
        input=text,
        model=model
    )

    return response.data[0].embedding

def retrieve_data(query, qdrant_client, k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01",
        query=query_embedding,
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []
    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload["average_rating"])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }

def process_context(context):
    
    formatted_context = ""

    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context

def build_prompt(preprocessed_context, question):

    prompt = f"""
    You are a shopping assistant that can answer questions about the products in stock.

    You will be given a question and a list of context.

    Instructions:
    - Answer the question based on the provided context only.
    - Never use word context and refer to it as the available products.
    - Do not use markdown formatting.

    Context:
    {preprocessed_context}

    Question:
    {question}
    """

    return prompt

def generate_answer(prompt):

    response = openai.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[{"role": "system", "content": prompt}],
        reasoning_effort="none"
    )

    return response.choices[0].message.content

def RAG_pipeline(question, qdrant_client, top_k=5):

    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    preprocess_context = process_context(retrieved_context)
    prompt = build_prompt(preprocess_context, question)
    answer = generate_answer(prompt)

    return answer