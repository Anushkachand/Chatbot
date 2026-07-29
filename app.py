
"""
ML Concepts Chatbot
--------------------
A Streamlit chat assistant that answers questions about Python, AI,
Machine Learning, Deep Learning, and Data Science using a TF-IDF +
Logistic Regression intent classifier.
 
Run with:
    streamlit run app.py
"""
 
import string
from datetime import datetime
 
import nltk
import pandas as pd
import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
 
# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
APP_NAME = "NeuraBot"
APP_TAGLINE = "Your AI, ML & Data Science companion"
MODEL_PATH = "chatbot_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATA_PATH = "chatbot.csv"
CONFIDENCE_THRESHOLD = 0.40
FALLBACK_MESSAGE = (
    "Hmm, I'm not confident about that one. Try asking me something about "
    "**Python, AI, Machine Learning, Deep Learning, or Data Science** 🙂"
)
 
st.set_page_config(
    page_title=f"{APP_NAME} | ML Chatbot",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)
 
# --------------------------------------------------------------------------
# Custom Styling
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef1f8 100%);
    }
    .app-header {
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
    }
    .app-header h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header p {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0.2rem;
    }
    .stChatMessage {
        border-radius: 14px;
    }
    .confidence-tag {
        display: inline-block;
        font-size: 0.72rem;
        color: #6b7280;
        margin-top: 4px;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
 
# --------------------------------------------------------------------------
# NLTK Setup (quiet, idempotent)
# --------------------------------------------------------------------------
def ensure_nltk_resources() -> None:
    """Download required NLTK corpora only if they aren't already present."""
    resources = {"tokenizers/punkt": "punkt", "corpora/stopwords": "stopwords"}
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)
 
 
ensure_nltk_resources()
 
# --------------------------------------------------------------------------
# Model & Data Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading intent model...")
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(
            f"Missing model file: `{e.filename}`. "
            "Make sure chatbot_model.pkl and vectorizer.pkl are in the app directory."
        )
        st.stop()
 
 
@st.cache_data(show_spinner="Loading knowledge base...")
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Missing dataset file: `{DATA_PATH}`.")
        st.stop()
 
 
model, vectorizer = load_model()
df = load_data()
 
# --------------------------------------------------------------------------
# Text Preprocessing
# --------------------------------------------------------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
 
 
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, tokenize, remove stopwords, and stem."""
    text = text.lower().translate(PUNCT_TABLE)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)
 
 
def chatbot_response(user_text: str) -> tuple[str, float]:
    """Return (response_text, confidence_score) for a given user message."""
    cleaned = clean_text(user_text)
    vector = vectorizer.transform([cleaned])
 
    intent = model.predict(vector)[0]
    confidence = float(model.predict_proba(vector).max())
 
    if confidence < CONFIDENCE_THRESHOLD:
        return FALLBACK_MESSAGE, confidence
 
    matches = df.loc[df["intent"] == intent, "response"]
    if matches.empty:
        return FALLBACK_MESSAGE, confidence
 
    return matches.iloc[0], confidence
 
 
# --------------------------------------------------------------------------
# Session State
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
 
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
 
# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <h1>🧠 {APP_NAME}</h1>
        <p>{APP_TAGLINE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# --------------------------------------------------------------------------
# Chat History
# --------------------------------------------------------------------------
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🧠"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        if message["role"] == "assistant" and "confidence" in message:
            st.markdown(
                f'<span class="confidence-tag">Confidence: {message["confidence"]:.0%} · {message["time"]}</span>',
                unsafe_allow_html=True,
            )
 
# --------------------------------------------------------------------------
# Chat Input
# --------------------------------------------------------------------------
user_input = st.chat_input("Ask about Python, AI, ML, or Data Science...")
 
if user_input:
    timestamp = datetime.now().strftime("%I:%M %p")
 
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_input)
 
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Thinking..."):
            answer, confidence = chatbot_response(user_input)
        st.write(answer)
        st.markdown(
            f'<span class="confidence-tag">Confidence: {confidence:.0%} · {timestamp}</span>',
            unsafe_allow_html=True,
        )
 
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "time": timestamp,
        }
    )
    st.session_state.total_queries += 1
 
# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## 🧠 {APP_NAME}")
    st.caption(APP_TAGLINE)
    st.divider()
 
    st.markdown("### ⚙️ Tech Stack")
    st.markdown(
        "- **NLP:** Tokenization, stopword removal, stemming\n"
        "- **Vectorization:** TF-IDF\n"
        "- **Model:** Logistic Regression\n"
        "- **Interface:** Streamlit"
    )
 
    st.divider()
    st.markdown("### 📊 Session Stats")
    st.metric("Questions Asked", st.session_state.total_queries)
 
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()
 
    st.divider()
    st.caption(f"© {datetime.now().year} {APP_NAME} · Built with Streamlit")