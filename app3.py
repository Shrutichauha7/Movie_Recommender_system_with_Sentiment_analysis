import streamlit as st
import pickle
import pandas as pd
import numpy as np
import random
import requests
from bs4 import BeautifulSoup


movies_df = pd.read_csv("movies.csv")

# ---------------- LOAD DATA ---------------- #

movies_dict = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Hybrid artifacts
movie_features = pickle.load(open(r"C:\Users\Lenovo\PycharmProjects\Movie_Recommender_system_with_Sentiment_analysis\movie_features.pkl","rb"))

movie_names = pickle.load(open(r"C:\Users\Lenovo\PycharmProjects\Movie_Recommender_system_with_Sentiment_analysis\movie_names.pkl", "rb"))


# ---------------- OCCUPATION MAP ---------------- #

occupation_map = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker",
    10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired",
    14: "sales/marketing", 15: "scientist", 16: "self-employed",
    17: "technician/engineer", 18: "tradesman/craftsman",
    19: "unemployed", 20: "writer"
}

# ---------------- CONTENT RECOMMENDER ---------------- #

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]] for i in movies_list]


# ---------------- CLUSTER RECOMMENDER ---------------- #

def recommend_from_cluster(cluster_id, top_n=5, genre=None):

    cluster_movies = movie_features[
        movie_features['cluster'] == cluster_id
    ].copy()

    if genre and genre in cluster_movies.columns:
        cluster_movies = cluster_movies[cluster_movies[genre] == 1]

    if cluster_movies.empty:
        return []

    cluster_movies['score'] = (
        cluster_movies['rating'] *
        np.log1p(cluster_movies['rating_count'])
    )

    top_movies = cluster_movies.sort_values(
        by='score', ascending=False
    ).head(top_n)

    top_movies['title'] = top_movies['movieId'].map(movie_names)

    return top_movies['title'].dropna().tolist()


# ---------------- HYBRID RECOMMENDER ---------------- #

def recommend_for_user(age, gender, occupation_name, genre=None):

    occupation = {v: k for k, v in occupation_map.items()}.get(occupation_name)

    if occupation is None:
        return []

    df = movies_df.copy()

    similar_users = df[
        (df['age'].between(age-5, age+5)) &
        (df['gender'] == gender) &
        (df['occupation'] == occupation)
    ]

    if genre and genre in df.columns:
        similar_users = similar_users[similar_users[genre] == 1]

    if similar_users.empty:
        return recommend_from_cluster(0, genre=genre)

    top_movies = similar_users.groupby('movieId')['rating'].agg(['mean','count'])

    top_movies['score'] = top_movies['mean'] * np.log1p(top_movies['count'])

    top_k = top_movies.sort_values(by='score', ascending=False).head(3).index

    clusters = movie_features[
        movie_features['movieId'].isin(top_k)
    ]['cluster']

    top_clusters = clusters.value_counts().head(2).index

    all_recs = []
    for c in top_clusters:
        all_recs.extend(recommend_from_cluster(c, top_n=3, genre=genre))

    return list(dict.fromkeys(all_recs))[:5]


# ---------------- REVIEWS ---------------- #

def fetch_reviews(imdb_id):
    url = f'https://www.imdb.com/title/{imdb_id}/reviews/'
    headers = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(url, headers=headers).text
    soup = BeautifulSoup(response, "html.parser")

    reviews = soup.find_all("div", class_="ipc-html-content-inner-div")

    return [r.text.strip() for r in reviews]


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

    return list(zip(reviews_sample, preds)), pos_percent, neg_percent


# ---------------- UI ---------------- #

st.title("🎬 Hybrid Movie Recommender System")

# Sidebar inputs
st.sidebar.header("👤 User Preferences")

age = st.sidebar.slider("Age", 10, 80, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender_val = 0 if gender == "Male" else 1

occupation_name = st.sidebar.selectbox(
    "Occupation",
    list(occupation_map.values())
)

genre_list = [
    'Action','Adventure','Animation',"Children's",'Comedy','Crime',
    'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',
    'Mystery','Romance','Sci-Fi','Thriller','War','Western'
]

genre = st.sidebar.selectbox("Preferred Genre", genre_list)

# Movie selection
selected_movie_name = st.selectbox(
    "Choose a movie",
    movies['title'].values
)

# Recommend button
if st.button("Recommend"):

    content_recs = recommend(selected_movie_name)

    hybrid_titles = recommend_for_user(
        age,
        gender_val,
        occupation_name,
        genre
    )

    hybrid_recs = movies[movies['title'].isin(hybrid_titles)]

    combined = pd.concat([pd.DataFrame(content_recs), hybrid_recs])
    combined = combined.drop_duplicates(subset='title')

    st.session_state.recommended_movies = combined.head(5).to_dict('records')


# Display results
if "recommended_movies" in st.session_state:

    st.subheader("🎯 Recommended Movies")

    cols = st.columns(5)

    for i in range(5):
        movie = pd.Series(st.session_state.recommended_movies[i])

        with cols[i]:
            st.image(movie.Poster_Url, use_container_width=True)
            st.caption(movie.title)