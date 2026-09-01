"""
system.py

Query layer for the customer-feedback RAG system.

Two things it adds beyond a plain "chat with your PDF" RAG:
  1. Metadata-filtered retrieval -- e.g. only search NEGATIVE feedback
     tagged "shipping and delivery" before generating an answer.
  2. Aggregate analytics -- sentiment/topic breakdowns for a dashboard,
     computed directly from the tagged data (not the LLM).
"""

import os
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# Loads GROQ_API_KEY from a local .env file (never commit that file).
load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-small-en"
LLM_MODEL = "openai/gpt-oss-20b"
INDEX_PATH = "faiss_index/feedback_index"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def query_feedback(question: str, sentiment: str = None, topic: str = None, k: int = 5) -> str:
    """
    Ask a natural-language question over the feedback corpus, optionally
    scoped to a sentiment (POSITIVE/NEGATIVE) and/or topic.

    Retrieval (embeddings + FAISS) stays on HuggingFace, which is stable.
    Answer generation goes through Groq instead of HuggingFace's serverless
    inference routing, since that free tier's model availability keeps
    shifting model-to-model.
    """
    vectorstore = load_vectorstore()

    filter_dict = {}
    if sentiment:
        filter_dict["sentiment"] = sentiment
    if topic:
        filter_dict["topic"] = topic

    docs = vectorstore.similarity_search(question, k=k, filter=filter_dict or None)

    context = "\n".join(f"- ({d.metadata['sentiment']}, {d.metadata['topic']}) {d.page_content}" for d in docs)
    prompt = (
        "You are analyzing customer feedback. Using only the feedback below, "
        f"answer the question concisely.\n\nFeedback:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return completion.choices[0].message.content


def sentiment_breakdown(tagged_csv: str = "tagged_feedback.csv") -> pd.DataFrame:
    """Aggregate sentiment counts per topic -- feeds the dashboard charts."""
    df = pd.read_csv(tagged_csv)
    return df.groupby(["topic", "sentiment"]).size().reset_index(name="count")


if __name__ == "__main__":
    print(sentiment_breakdown())
    answer = query_feedback(
        "What are customers unhappy about with shipping?",
        sentiment="NEGATIVE",
        topic="shipping and delivery",
    )
    print(answer)


