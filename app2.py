import streamlit as st
import pickle
import pandas as pd
import random
import requests
from bs4 import BeautifulSoup

# Load data
movies_dict = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# ---------------- RECOMMENDATION ---------------- #

def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended = []

    for i in movies_list:
        recommended.append(movies.iloc[i[0]])

    return recommended


# ---------------- FETCH REVIEWS ---------------- #


def fetch_reviews(imdb_id):
    url = f'https://www.imdb.com/title/{imdb_id}/reviews/'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'}


    response = requests.get(url, headers=headers).text

    soup = BeautifulSoup(response, "html.parser")

    reviews = soup.find_all("div", class_="ipc-html-content-inner-div")

    review_list = []

    for review in reviews:
        review_list.append(review.text.strip())

    return review_list

# ---------------- SENTIMENT ---------------- #

def predict_sentiment(reviews):

    if len(reviews) == 0:
        return [], 0, 0

    reviews_sample = random.sample(reviews, min(20, len(reviews)))

    X = vectorizer.transform(reviews_sample)

    preds = model.predict(X)

    positive = sum(preds)
    negative = len(preds) - positive

    pos_percent = round((positive / len(preds)) * 100, 2)
    neg_percent = round((negative / len(preds)) * 100, 2)

    results = list(zip(reviews_sample, preds))

    return results, pos_percent, neg_percent


# ---------------- SHOW REVIEWS ---------------- #

def show_reviews(movie):
    titleId = movie.imdb_id

    reviews = fetch_reviews(titleId)
    st.write("IMDb ID:", titleId)
    st.write("Fetched Reviews:", len(reviews))
    st.write("URL:", f"https://www.imdb.com/title/{titleId}/reviews")

    results, pos_percent, neg_percent = predict_sentiment(reviews)

    st.markdown("### 🎭 Audience Sentiment")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Positive Reviews: {pos_percent}%")

    with col2:
        st.error(f"Negative Reviews: {neg_percent}%")

    st.markdown("### 💬 Audience Reviews")

    for review, sentiment in results:

        if sentiment == 1:

            st.markdown(
                f"<div style='background:#1e7e34;padding:10px;border-radius:8px;margin-bottom:10px'>{review}</div>",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"<div style='background:#b02a37;padding:10px;border-radius:8px;margin-bottom:10px'>{review}</div>",
                unsafe_allow_html=True
            )


# ---------------- MOVIE DETAILS ---------------- #

def show_details(movie):

    st.markdown("## 🎬 Movie Details")

    col1, col2 = st.columns([1,2])

    with col1:
        st.image(movie.Poster_Url)

    with col2:
        st.subheader(movie.title)
        st.write("⭐ Rating:", movie.vote_average)
        st.write("🎬 Director:", ", ".join(movie.crew))
        st.write("🎭 Cast:", ", ".join(movie.cast[:3]))
        st.write("📝 Overview:")
        st.write(movie.original_overview)

        query = movie.title.replace(" ", "+") + "+official+trailer"

        youtube_link = f"https://www.youtube.com/results?search_query={query}"

        st.markdown(
            f"""
            <a href="{youtube_link}" target="_blank"
            style="
                display:inline-block;
                padding:6px 12px;
                font-size:13px;
                background-color:#ff4b4b;
                color:white;
                text-decoration:none;
                border-radius:6px;
                font-weight:500;
            ">
            ▶ Watch Trailer
            </a>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # SHOW REVIEWS
    show_reviews(movie)


# ---------------- UI ---------------- #

st.title("🎬 Movie Recommender System")

selected_movie_name = st.selectbox(
    "Enter a movie name",
    movies['title'].values
)


if st.button("Recommend"):
    st.session_state.recommended_movies = recommend(selected_movie_name)


if "recommended_movies" in st.session_state:

    recommended_movies = st.session_state.recommended_movies

    cols = st.columns(5)

    for i in range(5):

        movie = recommended_movies[i]

        with cols[i]:

            st.image(movie.Poster_Url, use_container_width=True)

            st.markdown(
                f"""
                            <div style="
                                height:55px;
                                text-align:center;
                                font-size:14px;
                                font-weight:600;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                            ">
                                {movie.title}
                            </div>
                            """,
                unsafe_allow_html=True
            )

            if st.button("View Details", key=f"details_{i}", use_container_width=True):
                st.session_state.selected_movie = movie


# SHOW DETAILS PAGE

if "selected_movie" in st.session_state:

    st.markdown("---")

    show_details(st.session_state.selected_movie)