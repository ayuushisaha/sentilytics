import pandas as pd
import streamlit as st
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("reviews.csv")

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)  # Fixed '@w+'
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["label"], test_size=0.2, random_state=42)

# ML Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('model', MultinomialNB())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# ---- Streamlit UI ----
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("📊 Sentiment Analysis on Movie Reviews")
st.write("Built with Python, NLTK, TF-IDF, Naive Bayes, and Streamlit")

st.sidebar.header("📌 Project Info")
st.sidebar.markdown("""
- Internship-Ready ML Project  
- Text Preprocessing  
- TF-IDF Vectorization  
- Naive Bayes Classification  
- Live Sentiment Prediction  
""")

# User Input
user_input = st.text_input("Enter a movie review:")
if st.button("Analyze Sentiment"):
    input_clean = clean_text(user_input)
    prediction = pipeline.predict([input_clean])[0]
    st.subheader(f"Predicted Sentiment: :blue[{prediction.upper()}]")

# Accuracy
st.subheader("Model Evaluation")
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{accuracy:.2f}")

with col2:
    st.write("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
    st.pyplot(fig)

# Classification Report
st.write("**Classification Report**")
st.json(report)
