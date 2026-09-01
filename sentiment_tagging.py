"""
sentiment_tagging.py

Tags each customer feedback row with:
  - sentiment: POSITIVE / NEGATIVE (via a HuggingFace sentiment model)
  - topic: zero-shot classification into a fixed set of business categories

This is the piece that turns the generic PDF-RAG project into a
customer-feedback-specific analytics tool: every chunk stored in the
vector index carries structured metadata (sentiment, topic) alongside
the raw text, so retrieval can be filtered ("show me negative feedback
about shipping") instead of pure semantic search.
"""

from transformers import pipeline
import pandas as pd

TOPICS = [
    "shipping and delivery",
    "product quality",
    "customer service",
    "pricing and billing",
    "app or website usability",
]


class FeedbackTagger:
    def __init__(self):
        # Lightweight, fast sentiment model
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        # Zero-shot topic classifier -- no need to fine-tune for each
        # new business category, just extend the TOPICS list.
        self.topic_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )

    def tag_sentiment(self, text: str) -> dict:
        result = self.sentiment_pipeline(text[:512])[0]
        return {"sentiment": result["label"], "sentiment_score": round(result["score"], 3)}

    def tag_topic(self, text: str) -> dict:
        result = self.topic_pipeline(text, candidate_labels=TOPICS)
        return {"topic": result["labels"][0], "topic_score": round(result["scores"][0], 3)}

    def tag_row(self, text: str) -> dict:
        return {**self.tag_sentiment(text), **self.tag_topic(text)}

    def tag_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        tags = df[text_col].apply(self.tag_row).apply(pd.Series)
        return pd.concat([df.reset_index(drop=True), tags], axis=1)


if __name__ == "__main__":
    df = pd.read_csv("sample_feedback.csv")
    tagger = FeedbackTagger()
    tagged = tagger.tag_dataframe(df)
    tagged.to_csv("tagged_feedback.csv", index=False)
    print(tagged[["feedback_id", "sentiment", "topic"]])
