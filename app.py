import streamlit as st
import pandas as pd
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download only stopwords
nltk.download("stopwords")

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ML Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    try:
        with open("chatbot_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


model, vectorizer = load_model()

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("chatbot.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()


df = load_data()

required_columns = {"intent", "response"}

if not required_columns.issubset(df.columns):
    st.error("chatbot.csv must contain 'intent' and 'response' columns.")
    st.stop()

# ---------------- TEXT PREPROCESSING ----------------

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):

    # lowercase
    text = text.lower()

    # remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # tokenize using split()
    words = text.split()

    # remove stopwords and stem
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ---------------- CHATBOT FUNCTION ----------------

def chatbot_response(user_text):

    cleaned = clean_text(user_text)

    vector = vectorizer.transform([cleaned])

    intent = model.predict(vector)[0]

    confidence = model.predict_proba(vector).max()

    if confidence < 0.40:
        return (
            "Sorry, I am not sure about that.\n\n"
            "Please ask something related to Python, AI, "
            "Machine Learning, Deep Learning, or Data Science."
        )

    response = df[df["intent"] == intent]

    if not response.empty:
        return response["response"].iloc[0]

    return "Sorry, I don't have an answer for that."


# ---------------- CHAT HISTORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- UI ----------------

st.title("🤖 Machine Learning Chatbot")

st.write(
    "Ask questions about Python, AI, Machine Learning, Deep Learning, NLP, and Data Science."
)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
user_input = st.chat_input("Type your question here...")

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    answer = chatbot_response(user_input)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.write(answer)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("About Chatbot")

    st.write("""
✅ NLP Text Processing

✅ TF-IDF Vectorization

✅ Logistic Regression

✅ Streamlit
""")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()