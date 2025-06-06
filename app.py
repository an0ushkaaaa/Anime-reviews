import streamlit as st
import requests
import time
import re
from transformers import pipeline
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# --------- Helper Functions ---------
def get_anime_id(anime_title):
    query = anime_title.replace(" ", "+")
    url = f"https://api.jikan.moe/v4/anime?q={query}&limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get("data", [])
        if data:
            return data[0]["mal_id"]
    return None

def get_reviews(anime_id, pages=5, filter_spoilers=True):
    reviews = []
    for page in range(1, pages + 1):
        url = f"https://api.jikan.moe/v4/anime/{anime_id}/reviews?page={page}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("data", [])
            for r in data:
                if filter_spoilers and r.get("is_spoiler", False):
                    continue
                review_text = r.get("review", "").replace('\n', ' ').strip()
                if len(review_text) > 50:
                    reviews.append({
                        "user": r["user"]["username"],
                        "score": r["score"],
                        "review": review_text
                    })
        else:
            print(f"Failed to fetch page {page}, status code: {response.status_code}")
        time.sleep(1)
    return reviews

def clean_review(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)
    return text.strip()

def chunk_reviews(reviews, chunk_size=3):
    chunks = []
    for i in range(0, len(reviews), chunk_size):
        chunk = " ".join(reviews[i:i + chunk_size])
        chunks.append(chunk[:1024])  # max chunk size limit
    return chunks

# --------- Transformers Pipelines ---------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
reflector = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-chat-hf",
    device=-1  # or device=0 if you have a GPU
)

def summarize_reviews(reviews, label="positive"):
    # Filter and clean reviews by sentiment label
    labeled = [r["review"] for r in reviews if r["sentiment_label"] == label]
    cleaned = [clean_review(r) for r in labeled if len(r) > 30]
    chunks = chunk_reviews(cleaned)
    
    summaries = [
        summarizer(chunk, max_length=80, min_length=30, do_sample=False)[0]["summary_text"]
        for chunk in chunks[:3]
    ]
    return "\n\n".join(summaries) if summaries else None

def reflect_on_summary(summary_text, sentiment_label):
    if not summary_text:
        return f"No {sentiment_label} reviews available to reflect on."
    prompt = f"<s>[INST] Provide a thoughtful reflection on the following {sentiment_label} summary:\n\n{summary_text} [/INST]</s>"
    output = reflector(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return output[0]["generated_text"]

# --------- Streamlit UI ---------
st.set_page_config(page_title="✨ Anime Review Summarizer & Reflector", layout="centered")
st.title("🌸 Anime Review Summarizer & Reflector")

anime_title = st.text_input("Enter Anime Title 🎥", "Naruto")

if st.button("Fetch, Summarize & Reflect 💬"):
    with st.spinner("Fetching and analyzing reviews..."):
        anime_id = get_anime_id(anime_title)
        if not anime_id:
            st.error("Anime not found. Please check the title and try again.")
        else:
            reviews = get_reviews(anime_id, pages=3)
            if not reviews:
                st.warning("No reviews found for this anime.")
            else:
                # Analyze sentiment for each review
                for r in reviews:
                    result = sentiment_pipeline(r["review"][:512])[0]
                    r["sentiment_label"] = result["label"].lower()

                # Summarize positive and negative reviews
                pos_summary = summarize_reviews(reviews, label="positive")
                neg_summary = summarize_reviews(reviews, label="negative")

                # Reflect on summaries
                pos_reflection = reflect_on_summary(pos_summary, "positive")
                neg_reflection = reflect_on_summary(neg_summary, "negative")

                # Display results
                st.subheader("💖 Reflection on Positive Reviews")
                st.write(pos_reflection)

                st.subheader("💢 Reflection on Negative Reviews")
                st.write(neg_reflection)

