import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
import string
import random
import time

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="IntelliBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Load Files
# ----------------------------------------------------
model = pickle.load(open("chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
df = pd.read_csv("chatbot.csv")

# ----------------------------------------------------
# NLP
# ----------------------------------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------
# st.markdown("""
# <style>

# #MainMenu{
# visibility:hidden;
# }

# footer{
# visibility:hidden;
# }

# header{
# visibility:hidden;
# }

# .stApp{

# background:linear-gradient(135deg,#0F172A,#1E293B,#334155);

# color:white;

# }

# .big-title{

# text-align:center;

# font-size:48px;

# font-weight:bold;

# color:#60A5FA;

# }

# .sub-title{

# text-align:center;

# font-size:18px;

# color:#CBD5E1;

# margin-bottom:25px;

# }

# .user-card{

# background:#2563EB;

# padding:15px;

# border-radius:15px;

# margin-top:10px;

# margin-bottom:10px;

# font-size:18px;

# color:white;

# box-shadow:0px 4px 12px rgba(0,0,0,.3);

# }

# .bot-card{

# background:#14B8A6;

# padding:15px;

# border-radius:15px;

# margin-top:10px;

# margin-bottom:10px;

# font-size:18px;

# color:white;

# box-shadow:0px 4px 12px rgba(0,0,0,.3);

# }

# .metric{

# background:#1E293B;

# padding:18px;

# border-radius:15px;

# text-align:center;

# color:white;

# }

# .stButton>button{

# width:100%;

# height:45px;

# border-radius:10px;

# font-size:17px;

# background:#2563EB;

# color:white;

# }

# </style>
# """, unsafe_allow_html=True)




# ----------------------------------------------------
# Custom CSS (Light Theme)
# ----------------------------------------------------
st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}


/* Main Background */

.stApp{

background:linear-gradient(
135deg,
#F8FAFC,
#E2E8F0,
#CBD5E1
);

color:#1E293B;

}


/* Main Title */

.big-title{

text-align:center;

font-size:48px;

font-weight:700;

color:#2563EB;

}


/* Subtitle */

.sub-title{

text-align:center;

font-size:18px;

color:#475569;

margin-bottom:25px;

}


/* User Message Card */

.user-card{

background:#DBEAFE;

padding:15px;

border-radius:15px;

margin-top:10px;

margin-bottom:10px;

font-size:18px;

color:#1E3A8A;

box-shadow:
0px 4px 12px rgba(0,0,0,0.15);

}


/* Bot Message Card */

.bot-card{

background:#CCFBF1;

padding:15px;

border-radius:15px;

margin-top:10px;

margin-bottom:10px;

font-size:18px;

color:#134E4A;

box-shadow:
0px 4px 12px rgba(0,0,0,0.15);

}


/* Metrics */

.metric{

background:white;

padding:18px;

border-radius:15px;

text-align:center;

color:#1E293B;

box-shadow:
0px 4px 10px rgba(0,0,0,0.10);

}


/* Streamlit Sidebar */

section[data-testid="stSidebar"]{

background:#FFFFFF;

}


section[data-testid="stSidebar"] *{

color:#1E293B !important;

}


/* Buttons */

.stButton>button{

width:100%;

height:45px;

border-radius:10px;

font-size:17px;

background:#2563EB;

color:white;

border:none;

font-weight:600;

}


.stButton>button:hover{

background:#1D4ED8;

color:white;

}


/* Chat Container */

[data-testid="stChatMessage"]{

background:#FFFFFF;

border-radius:15px;

padding:12px;

margin-bottom:10px;

box-shadow:
0px 3px 10px rgba(0,0,0,0.12);

}


/* Text Color Fix */

h1,h2,h3,h4,h5,h6,p,span,label{

color:#1E293B !important;

}


/* Input Box */

[data-testid="stChatInput"] textarea{

background:white !important;

color:#1E293B !important;

border-radius:12px;

}


/* Info Boxes */

.stAlert{

background:#FFFFFF;

color:#1E293B;

border-radius:12px;

}


/* Download Button */

.stDownloadButton button{

background:#14B8A6;

color:white;

border-radius:10px;

font-size:16px;

font-weight:bold;

}


</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712027.png",
        width=120
    )

    st.title("🤖 IntelliBot AI")

    st.caption("Machine Learning Chatbot")

    st.divider()

    st.subheader("Features")

    st.success("✔ NLP")

    st.success("✔ Machine Learning")

    st.success("✔ TF-IDF")

    st.success("✔ Logistic Regression")

    st.success("✔ Fast Response")

    st.divider()

    st.subheader("Statistics")

    st.metric("Questions", len(df))

    st.metric("Intents", df["intent"].nunique())

    st.metric("Model", "Logistic Regression")

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages=[]

        st.rerun()

# ----------------------------------------------------
# Header
# ----------------------------------------------------
st.markdown(
    "<div class='big-title'>🤖 IntelliBot AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Your Personal AI Assistant</div>",
    unsafe_allow_html=True
)

# ----------------------------------------------------
# Text Cleaning Function
# ----------------------------------------------------
def clean_text(text):
    """
    Clean user input before prediction
    """
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords and stem
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ----------------------------------------------------
# Response Generator
# ----------------------------------------------------
def get_response(user_input):

    # Clean text
    cleaned_text = clean_text(user_input)

    # Vectorize
    vector = vectorizer.transform([cleaned_text])

    # Predict Intent
    predicted_intent = model.predict(vector)[0]

    # Prediction confidence
    confidence = None

    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(vector).max())

    # Find all matching responses
    responses = df[df["intent"] == predicted_intent]["response"].tolist()

    # Default response
    if len(responses) == 0:
        return (
            "I'm sorry, I couldn't understand your question.",
            predicted_intent,
            confidence
        )

    # Random response (if multiple responses exist)
    response = random.choice(responses)

    return response, predicted_intent, confidence


# ----------------------------------------------------
# Session State
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0


# ----------------------------------------------------
# Welcome Screen
# ----------------------------------------------------
if len(st.session_state.messages) == 0:

    st.markdown("## 👋 Welcome!")

    st.write(
        "Ask me anything about programming, AI, machine learning, "
        "web development, resumes, interviews, and more."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("🐍 Python")

        st.info("☕ Java")

        st.info("🌐 HTML")

    with c2:
        st.info("🎨 CSS")

        st.info("⚡ JavaScript")

        st.info("🗄 SQL")

    with c3:
        st.info("🤖 AI")

        st.info("📊 Machine Learning")

        st.info("📄 Resume")


# ----------------------------------------------------
# Dashboard Metrics
# ----------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💬 Messages", len(st.session_state.messages))

with col2:
    st.metric("📚 Intents", df["intent"].nunique())

with col3:
    st.metric("📄 Dataset", len(df))


    # ----------------------------------------------------
# Display Chat History
# ----------------------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Show confidence if available
        if (
            message["role"] == "assistant"
            and "confidence" in message
            and message["confidence"] is not None
        ):
            st.progress(message["confidence"])
            st.caption(
                f"Confidence : {message['confidence']*100:.1f}%"
            )

# ----------------------------------------------------
# Chat Input
# ----------------------------------------------------
user_input = st.chat_input("💬 Ask me anything...")

if user_input:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Bot response
    with st.chat_message("assistant"):

        # Typing animation
        with st.spinner("🤖 Thinking..."):
            time.sleep(1)

            response, intent, confidence = get_response(user_input)

        st.markdown(response)

        # Confidence bar
        if confidence is not None:
            st.progress(confidence)
            st.caption(
                f"Prediction Confidence : {confidence*100:.1f}%"
            )

        # Show detected intent
        st.info(f"Detected Intent : **{intent}**")

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "intent": intent,
            "confidence": confidence
        }
    )

    st.session_state.chat_count += 1

# ----------------------------------------------------
# Chat Analytics
# ----------------------------------------------------
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "💬 Total Messages",
        len(st.session_state.messages)
    )

with col2:
    st.metric(
        "🤖 Bot Replies",
        len(
            [
                m for m in st.session_state.messages
                if m["role"] == "assistant"
            ]
        )
    )

with col3:
    st.metric(
        "📊 Questions Asked",
        st.session_state.chat_count
    )

    # ----------------------------------------------------
# Suggested Questions
# ----------------------------------------------------
st.divider()

st.subheader("💡 Try asking me")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🐍 What is Python?")
    st.info("🌐 Explain HTML")
    st.info("🎨 What is CSS?")

with col2:
    st.info("⚡ Explain JavaScript")
    st.info("🗄 What is SQL?")
    st.info("🤖 What is AI?")

with col3:
    st.info("📊 What is Machine Learning?")
    st.info("📄 Resume Tips")
    st.info("💼 Interview Questions")

# ----------------------------------------------------
# Download Chat
# ----------------------------------------------------
st.divider()

chat_history = ""

for msg in st.session_state.messages:
    role = "You" if msg["role"] == "user" else "Bot"
    chat_history += f"{role}: {msg['content']}\n\n"

st.download_button(
    label="📥 Download Chat",
    data=chat_history,
    file_name="chat_history.txt",
    mime="text/plain"
)

# ----------------------------------------------------
# Clear Chat
# ----------------------------------------------------
if st.button("🗑️ Clear Conversation"):

    st.session_state.messages = []
    st.session_state.chat_count = 0

    st.success("Conversation Cleared!")

    time.sleep(1)

    st.rerun()

# ----------------------------------------------------
# About Section
# ----------------------------------------------------
st.divider()

with st.expander("ℹ About IntelliBot AI"):

    st.write("""
### 🤖 IntelliBot AI

IntelliBot AI is a Machine Learning based chatbot developed using:

- 🐍 Python
- 🎈 Streamlit
- 🤖 Scikit-learn
- 📊 TF-IDF Vectorizer
- 🧠 Logistic Regression
- 📚 NLTK

Features:

✔ Natural Language Processing

✔ Intent Detection

✔ Fast Response

✔ Chat History

✔ Download Conversation

✔ Modern User Interface

✔ Machine Learning Prediction
""")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.divider()

st.markdown(
"""
<div style='text-align:center;
padding:20px;
color:gray;'>

<h4>🤖 IntelliBot AI</h4>

Developed by <b>Ananya Gupta</b>

B.Tech Computer Science & Engineering

Machine Learning | NLP | Streamlit

© 2026 All Rights Reserved

</div>
""",
unsafe_allow_html=True
)