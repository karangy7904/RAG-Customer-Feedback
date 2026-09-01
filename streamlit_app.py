import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq

from sentiment_tagging import FeedbackTagger, TOPICS
from ingest_feedback import build_documents
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================

# STREAMLIT PAGE CONFIG

# ============================================================

st.set_page_config(
page_title="Customer Feedback RAG",
layout="wide"
)

st.title("📊 Customer Feedback Analysis & Q&A")

# ============================================================

# CACHED RESOURCES

# ============================================================

@st.cache_resource
def load_embeddings():
"""
Load the Hugging Face embedding model once
and reuse it across Streamlit reruns.
"""
return HuggingFaceEmbeddings(
model_name="BAAI/bge-small-en"
)

@st.cache_resource
def load_tagger():
"""
Load the feedback tagger once.
"""
return FeedbackTagger()

@st.cache_resource
def get_groq_client():
"""
Create the Groq client using Streamlit Secrets.
"""
return Groq(
api_key=st.secrets["GROQ_API_KEY"]
)

# ============================================================

# SESSION STATE

# ============================================================

if "tagged_df" not in st.session_state:
st.session_state.tagged_df = None

if "vectorstore" not in st.session_state:
st.session_state.vectorstore = None

# ============================================================

# FILE UPLOAD

# ============================================================

uploaded_file = st.file_uploader(
"Upload customer feedback (CSV with a 'text' column)",
type="csv"
)

# ============================================================

# PROCESS FEEDBACK

# ============================================================

if uploaded_file and st.button("Process feedback"):

```
# --------------------------------------------------------
# Read CSV
# --------------------------------------------------------

with st.spinner("Reading feedback data..."):

    df = pd.read_csv(uploaded_file)


# --------------------------------------------------------
# Validate CSV
# --------------------------------------------------------

if "text" not in df.columns:

    st.error(
        "The uploaded CSV must contain a column named 'text'."
    )

    st.stop()


if df.empty:

    st.error(
        "The uploaded CSV is empty."
    )

    st.stop()


# --------------------------------------------------------
# Tag sentiment and topic
# --------------------------------------------------------

with st.spinner(
    "Tagging sentiment and topic for each row..."
):

    tagger = load_tagger()

    tagged_df = tagger.tag_dataframe(df)

    st.session_state.tagged_df = tagged_df


# --------------------------------------------------------
# Build searchable vector index
# --------------------------------------------------------

with st.spinner(
    "Building searchable index..."
):

    embeddings = load_embeddings()

    docs = build_documents(tagged_df)

    st.session_state.vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )


st.success(
    f"Processed {len(tagged_df)} feedback entries."
)
```

# ============================================================

# DASHBOARD

# ============================================================

if st.session_state.tagged_df is not None:

```
df = st.session_state.tagged_df


# --------------------------------------------------------
# Dashboard columns
# --------------------------------------------------------

col1, col2 = st.columns(2)


# ========================================================
# OVERALL SENTIMENT
# ========================================================

with col1:

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "sentiment",
        "count"
    ]

    fig1 = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        title="Overall Sentiment"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )


# ========================================================
# SENTIMENT BY TOPIC
# ========================================================

with col2:

    topic_sentiment = (
        df.groupby(
            ["topic", "sentiment"]
        )
        .size()
        .reset_index(name="count")
    )

    fig2 = px.bar(
        topic_sentiment,
        x="topic",
        y="count",
        color="sentiment",
        barmode="group",
        title="Sentiment by Topic"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ========================================================
# QUESTION ANSWERING
# ========================================================

st.divider()

st.subheader(
    "Ask a question about the feedback"
)


# --------------------------------------------------------
# Filters
# --------------------------------------------------------

fc1, fc2 = st.columns(2)


with fc1:

    sentiment_filter = st.selectbox(
        "Filter by sentiment",
        [
            "Any",
            "POSITIVE",
            "NEGATIVE"
        ]
    )


with fc2:

    topic_filter = st.selectbox(
        "Filter by topic",
        ["Any"] + TOPICS
    )


# --------------------------------------------------------
# Question input
# --------------------------------------------------------

question = st.text_input(
    "Your question",
    placeholder=(
        "What are customers unhappy about with shipping?"
    )
)


# ========================================================
# ASK QUESTION
# ========================================================

if question and st.button("Ask"):

    # ----------------------------------------------------
    # Make sure vectorstore exists
    # ----------------------------------------------------

    if st.session_state.vectorstore is None:

        st.error(
            "Please process the feedback before asking a question."
        )

        st.stop()


    # ----------------------------------------------------
    # Build metadata filters
    # ----------------------------------------------------

    filter_dict = {}


    if sentiment_filter != "Any":

        filter_dict["sentiment"] = sentiment_filter


    if topic_filter != "Any":

        filter_dict["topic"] = topic_filter


    # ----------------------------------------------------
    # Retrieve relevant documents
    # ----------------------------------------------------

    with st.spinner(
        "Retrieving and generating answer..."
    ):

        docs = (
            st.session_state.vectorstore
            .similarity_search(
                question,
                k=5,
                filter=filter_dict or None
            )
        )


        # ------------------------------------------------
        # No results
        # ------------------------------------------------

        if not docs:

            st.warning(
                "No feedback matched your selected filters."
            )

            st.stop()


        # ------------------------------------------------
        # Build context
        # ------------------------------------------------

        context = "\n".join(
            f"- "
            f"({d.metadata['sentiment']}, "
            f"{d.metadata['topic']}) "
            f"{d.page_content}"
            for d in docs
        )


        # ------------------------------------------------
        # Prompt
        # ------------------------------------------------

        prompt = (
            "You are analyzing customer feedback. "
            "Using only the feedback below, "
            "answer the question concisely.\n\n"
            f"Feedback:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )


        # ------------------------------------------------
        # Groq client
        # ------------------------------------------------

        client = get_groq_client()


        # ------------------------------------------------
        # Generate answer
        # ------------------------------------------------

        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0.3
        )


        answer = (
            completion
            .choices[0]
            .message
            .content
        )


    # ====================================================
    # DISPLAY ANSWER
    # ====================================================

    st.write(answer)


    # ====================================================
    # SOURCE DOCUMENTS
    # ====================================================

    with st.expander(
        "Source feedback used"
    ):

        for doc in docs:

            st.write(
                f"**{doc.metadata['sentiment']} / "
                f"{doc.metadata['topic']}** — "
                f"{doc.page_content}"
            )
```
