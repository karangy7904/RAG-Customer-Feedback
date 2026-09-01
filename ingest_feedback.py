"""
ingest_feedback.py

Loads raw customer feedback (CSV), tags each row with sentiment + topic,
embeds the text with a HuggingFace embedding model, and stores it in a
FAISS index with the tags attached as metadata. This is what enables
*filtered* retrieval later (system.py), not just plain semantic search.
"""

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from sentiment_tagging import FeedbackTagger

EMBEDDING_MODEL = "BAAI/bge-small-en"
INDEX_PATH = "faiss_index/feedback_index"


def build_documents(tagged_df: pd.DataFrame) -> list[Document]:
    docs = []
    for _, row in tagged_df.iterrows():
        docs.append(
            Document(
                page_content=row["text"],
                metadata={
                    "feedback_id": row["feedback_id"],
                    "customer": row["customer"],
                    "date": row["date"],
                    "channel": row["channel"],
                    "sentiment": row["sentiment"],
                    "topic": row["topic"],
                },
            )
        )
    return docs


def ingest(csv_path: str = "sample_feedback.csv"):
    df = pd.read_csv(csv_path)

    print(f"Tagging {len(df)} feedback rows with sentiment + topic...")
    tagger = FeedbackTagger()
    tagged_df = tagger.tag_dataframe(df)

    print("Embedding and building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docs = build_documents(tagged_df)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_PATH)

    tagged_df.to_csv("tagged_feedback.csv", index=False)
    print(f"Done. Index saved to {INDEX_PATH}, tagged data saved to tagged_feedback.csv")
    return tagged_df


if __name__ == "__main__":
    ingest()
