# 🌸 Anime-reviews
This is a Streamlit web app that fetches anime reviews from the Jikan API (an unofficial MyAnimeList API), summarizes them using OpenAI’s GPT models, and displays concise review summaries for easy consumption.

**👋 How to use it**

Streamlit App Link: https://anime-reviews-79ol5fhwwvpcbnzcoaqwn9.streamlit.app/

1. Enter an anime name
2. Fetch the summary of reviews

That's it!

⚙️ **How it works**

This project uses the Jikan API to fetch reviews of an anime based on its title and MAL ID. Once the reviews are retrieved (up to ~125 reviews across 5 pages), they are assigned a sentiment(Positive, Negative) using the DISTIL BERT model and summarized using the Facebook BART model.

To go beyond simple summarization, the tool then uses the OpenAI API to generate a thoughtful reflection on the reviews, offering more human-like insights that might be useful for users.

The entire application is built and deployed using Streamlit for a seamless and interactive experience.

🤖 **AI Reflection**

ChatGPT was a huge part of this project, not only for generating text within the application but also in helping write and structure the code. It suggested suitable models from Hugging Face, and guided me through deploying the app using Streamlit.

That said, I often had to manually adapt and debug the code, especially when differences between environments (like Kaggle vs. Streamlit Cloud) caused unexpected issues. For example, the OpenAI API worked well in Kaggle but responded slowly on Streamlit, prompting me to troubleshoot based on concepts learned during class. These hands-on debugging sessions deepened my understanding of both ML pipelines and deployment nuances.

📸 **Screenshots**
<img width="1469" alt="Screenshot 2025-06-06 at 8 48 22 PM" src="https://github.com/user-attachments/assets/471835e1-2fb3-4ead-be3b-c514e5b27e05" />


<img width="1470" alt="Screenshot 2025-06-06 at 3 18 26 PM" src="https://github.com/user-attachments/assets/77381d00-af31-429a-b3a8-5c28d2efb158" />
