import streamlit as st

st.set_page_config(page_title="✨ Anime Review Summarizer", layout="centered")
st.title("🌸 Anime Review Summarizer")

anime_title = st.text_input("Enter Anime Title 🎥", "Naruto")

if st.button("Fetch and Summarize Reviews 💬"):
    with st.spinner("Processing..."):
        anime_id = get_anime_id(anime_title)
        reviews = get_reviews(anime_id, pages=3)
        
        # your sentiment code here
        # your summarizer code here

        st.subheader("Positive Summary 💖")
        st.write(positive_summary)

        st.subheader("Negative Summary 💢")
        st.write(negative_summary)
