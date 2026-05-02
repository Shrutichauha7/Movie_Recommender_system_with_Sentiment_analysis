import streamlit as st
import pickle
import pandas as pd
import random
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests

# ---------------- LOAD DATA ---------------- #

movies_dict = pickle.load(open(r'C:\Users\Lenovo\PycharmProjects\Movie_Recommender_system_with_Sentiment_analysis\movies.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- USER INPUT ---------------- #

st.sidebar.header(" User Preferences")

age = st.sidebar.slider("Age", 10, 80, 25)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender_val = 0 if gender == "Male" else 1

occupation_list = [
    "other","academic/educator","artist","clerical/admin",
    "college/grad student","customer service","doctor/health care",
    "executive/managerial","farmer","homemaker",
    "K-12 student","lawyer","programmer","retired",
    "sales/marketing","scientist","self-employed",
    "technician/engineer","tradesman/craftsman",
    "unemployed","writer"
]

occupation = st.sidebar.selectbox("Occupation", occupation_list)

genre_list = [
    'Action','Adventure','Animation',"Children's",'Comedy','Crime',
    'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',
    'Mystery','Romance','Sci-Fi','Thriller','War','Western'
]

genre = st.sidebar.selectbox("Preferred Genre", genre_list)

# ---------------- USER-BASED RECOMMENDER ---------------- #

def recommend_for_user(age, gender, occupation, genre):

    df = movies.copy()

    # Simulated personalization (since no rating matrix available)
    filtered = df.copy()

    # genre filtering (if exists)
    if 'genres' in df.columns:
        filtered = filtered[filtered['genres'].str.contains(genre, case=False, na=False)]

    # fallback if nothing matches
    if filtered.empty:
        filtered = df.sample(10)

    # add randomness for diversity
    return filtered.sample(min(5, len(filtered))).to_dict('records')

# ---------------- REVIEWS ---------------- #




def predict_sentiment(reviews):

    if len(reviews) == 0:
        return [], 0, 0

    sample = random.sample(reviews, min(20, len(reviews)))

    X = vectorizer.transform(sample)
    preds = model.predict(X)

    pos = sum(preds)
    neg = len(preds) - pos

    return list(zip(sample, preds)), round(pos/len(preds)*100,2), round(neg/len(preds)*100,2)

# ---------------- DETAILS ---------------- #

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

        query = movie.title.replace(" ", "+") + "+trailer"
        url = f"https://www.youtube.com/results?search_query={query}"

        st.markdown(f"[▶ Watch Trailer]({url})")

    st.markdown("---")




# ---------------- UI ---------------- #

st.title("Movie Recommender System")




# ---------------- RECOMMEND ---------------- #

if st.button("Recommend"):

    st.session_state.recommended_movies = recommend_for_user(
        age, gender_val, occupation, genre
    )

# ---------------- DISPLAY ---------------- #

if "recommended_movies" in st.session_state:

    recs = st.session_state.recommended_movies

    if len(recs) == 0:
        st.warning("No recommendations found.")
    else:

        cols = st.columns(min(5, len(recs)))

        for i, movie in enumerate(recs[:5]):

            with cols[i]:

                st.image(movie["Poster_Url"], use_container_width=True)

                st.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        font-size:14px;
                        font-weight:600;
                        height:50px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                    ">
                        {movie["title"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button("View Details", key=f"btn_{i}"):

                    class Obj:
                        pass

                    m = Obj()
                    for k,v in movie.items():
                        setattr(m, k, v)

                    st.session_state.selected_movie = m

# ---------------- DETAILS PAGE ---------------- #

if "selected_movie" in st.session_state:
    st.markdown("---")
    show_details(st.session_state.selected_movie)