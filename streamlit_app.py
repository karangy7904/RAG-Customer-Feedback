"""
streamlit_app.py

Front-end for the customer-feedback RAG tool.

Adds, on top of the original "upload a PDF and chat with it" UI:
  - A CSV upload path for structured feedback data
  - A live sentiment/topic dashboard (Plotly)
  - Sidebar filters so the chat only retrieves feedback matching a
    chosen sentiment and/or topic before the LLM answers
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from groq import Groq

from sentiment_tagging import FeedbackTagger, TOPICS
from ingest_feedback import build_documents
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# Loads HUGGINGFACEHUB_API_TOKEN from a local .env file (never commit that file).
load_dotenv()

st.set_page_config(page_title="Customer Feedback RAG", layout="wide")
st.title("📊 Customer Feedback Analysis & Q&A")

if "tagged_df" not in st.session_state:
    st.session_state.tagged_df = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload customer feedback (CSV with a 'text' column)", type="csv")

if uploaded_file and st.button("Process feedback"):
    with st.spinner("Tagging sentiment and topic for each row..."):
        df = pd.read_csv(uploaded_file)
        tagger = FeedbackTagger()
        tagged_df = tagger.tag_dataframe(df)
        st.session_state.tagged_df = tagged_df

    with st.spinner("Building searchable index..."):
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        docs = build_documents(tagged_df)
        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

    st.success(f"Processed {len(tagged_df)} feedback entries.")

if st.session_state.tagged_df is not None:
    df = st.session_state.tagged_df

    col1, col2 = st.columns(2)
    with col1:
        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        fig1 = px.pie(sentiment_counts, names="sentiment", values="count", title="Overall Sentiment")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        topic_sentiment = df.groupby(["topic", "sentiment"]).size().reset_index(name="count")
        fig2 = px.bar(
            topic_sentiment, x="topic", y="count", color="sentiment",
            barmode="group", title="Sentiment by Topic"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Ask a question about the feedback")

    fc1, fc2 = st.columns(2)
    with fc1:
        sentiment_filter = st.selectbox("Filter by sentiment", ["Any", "POSITIVE", "NEGATIVE"])
    with fc2:
        topic_filter = st.selectbox("Filter by topic", ["Any"] + TOPICS)

    question = st.text_input("Your question", placeholder="What are customers unhappy about with shipping?")

    if question and st.button("Ask"):
        filter_dict = {}
        if sentiment_filter != "Any":
            filter_dict["sentiment"] = sentiment_filter
        if topic_filter != "Any":
            filter_dict["topic"] = topic_filter

        with st.spinner("Retrieving and generating answer..."):
            docs = st.session_state.vectorstore.similarity_search(
                question, k=5, filter=filter_dict or None
            )
            context = "\n".join(
                f"- ({d.metadata['sentiment']}, {d.metadata['topic']}) {d.page_content}" for d in docs
            )
            prompt = (
                "You are analyzing customer feedback. Using only the feedback below, "
                f"answer the question concisely.\n\nFeedback:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            answer = completion.choices[0].message.content

        st.write(answer)
        with st.expander("Source feedback used"):
            for doc in docs:
                st.write(f"**{doc.metadata['sentiment']} / {doc.metadata['topic']}** — {doc.page_content}")
