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

        # Lightweight sentiment model
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )

        # Smaller zero-shot model than facebook/bart-large-mnli
        self.topic_pipeline = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1",
            device=-1
        )

    def tag_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        batch_size: int = 8
    ) -> pd.DataFrame:

        texts = (
            df[text_col]
            .fillna("")
            .astype(str)
            .str.slice(0, 512)
            .tolist()
        )

        # ------------------------------------------
        # SENTIMENT BATCH INFERENCE
        # ------------------------------------------

        sentiment_results = []

        for i in range(0, len(texts), batch_size):

            batch = texts[i:i + batch_size]

            results = self.sentiment_pipeline(
                batch,
                batch_size=batch_size,
                truncation=True,
                max_length=512
            )

            sentiment_results.extend(results)

        # ------------------------------------------
        # TOPIC BATCH INFERENCE
        # ------------------------------------------

        topic_results = []

        for text in texts:

            result = self.topic_pipeline(
                text,
                candidate_labels=TOPICS
            )

            topic_results.append(result)

        # ------------------------------------------
        # CREATE TAG COLUMNS
        # ------------------------------------------

        sentiments = [
            result["label"]
            for result in sentiment_results
        ]

        sentiment_scores = [
            round(result["score"], 3)
            for result in sentiment_results
        ]

        topics = [
            result["labels"][0]
            for result in topic_results
        ]

        topic_scores = [
            round(result["scores"][0], 3)
            for result in topic_results
        ]

        # ------------------------------------------
        # BUILD OUTPUT DATAFRAME
        # ------------------------------------------

        result_df = df.reset_index(drop=True).copy()

        result_df["sentiment"] = sentiments
        result_df["sentiment_score"] = sentiment_scores
        result_df["topic"] = topics
        result_df["topic_score"] = topic_scores

        return result_df


# --------------------------------------------------
# LOCAL TESTING
# --------------------------------------------------

if __name__ == "__main__":

    df = pd.read_csv(
        "sample_feedback.csv"
    )

    tagger = FeedbackTagger()

    tagged = tagger.tag_dataframe(
        df,
        batch_size=8
    )

    tagged.to_csv(
        "tagged_feedback.csv",
        index=False
    )

    print(
        tagged[
            [
                "feedback_id",
                "sentiment",
                "topic"
            ]
        ]
    )

