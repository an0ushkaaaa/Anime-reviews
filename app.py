import requests
import time
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

# ----- Data fetching -----
def get_anime_id(anime_title):
    query = anime_title.replace(" ", "+")
    url = f"https://api.jikan.moe/v4/anime?q={query}&limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get("data", [])
        if data:
            return data[0]["mal_id"]
    return None

def get_reviews(anime_id, pages=3, filter_spoilers=True):
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
            print(f"Failed to fetch page {page}")
        time.sleep(1)
    return reviews

# ----- Sentiment analysis -----
sentiment_pipeline = pipeline("sentiment-analysis")

def label_sentiments(reviews):
    for r in reviews:
        result = sentiment_pipeline(r["review"][:512])[0]
        r["sentiment_label"] = result["label"].lower()
    return reviews

# ----- Cleaning and chunking -----
def clean_review(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)
    return text.strip()

def chunk_reviews(reviews, chunk_size=3):
    chunks = []
    for i in range(0, len(reviews), chunk_size):
        chunk = " ".join(reviews[i:i + chunk_size])
        chunks.append(chunk[:1024])
    return chunks

# ----- Summarization -----
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_chunks(chunks):
    summaries = [summarizer(chunk, max_length=80, min_length=30, do_sample=False)[0]["summary_text"]
                 for chunk in chunks[:3]]
    return summaries

# ----- Paraphrasing -----
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

def rephrase_text(text):
    input_text = f"paraphrase: {text} </s>"
    encoding = tokenizer.encode_plus(input_text, padding=True, return_tensors="pt")
    outputs = model.generate(
        **encoding,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
        temperature=1.5,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.set_page_config(page_title="✨ Anime Review Summarizer", layout="centered")
st.title("🌸 Anime Review Summarizer")

anime_title = st.text_input("Enter Anime Title 🎥", "Naruto")

if st.button("Fetch and Summarize Reviews 💬"):
    with st.spinner("Fetching reviews and analyzing..."):
        anime_id = get_anime_id(anime_title)
        if anime_id is None:
            st.error(f"Could not find anime titled '{anime_title}'")
        else:
            reviews = get_reviews(anime_id, pages=3)
            reviews = label_sentiments(reviews)

            positive_reviews = [r["review"] for r in reviews if r["sentiment_label"] == "positive"]
            negative_reviews = [r["review"] for r in reviews if r["sentiment_label"] == "negative"]

            cleaned_positive = [clean_review(r) for r in positive_reviews if len(r) > 30]
            cleaned_negative = [clean_review(r) for r in negative_reviews if len(r) > 30]

            positive_chunks = chunk_reviews(cleaned_positive)
            negative_chunks = chunk_reviews(cleaned_negative)

            positive_summaries = summarize_chunks(positive_chunks)
            negative_summaries = summarize_chunks(negative_chunks)

            polished_positive = rephrase_text(" ".join(positive_summaries))
            polished_negative = rephrase_text(" ".join(negative_summaries))

            st.subheader("🟢 Positive Summary")
            st.write(polished_positive)

            st.subheader("🔴 Negative Summary")
            st.write(polished_negative)

