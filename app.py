import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Load Data
# =========================

movies = pd.read_csv("data/ml-latest-small/movies.csv")
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")


# =========================
# Page Setup
# =========================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite movie and get personalized recommendations.")


# =========================
# Content-Based Filtering
# =========================

movies["genres_clean"] = movies["genres"].str.replace(
    "|", " ", regex=False
)

tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(movies["genres_clean"])


def recommend_movies(movie_title, n=5):

    movie_index = movies.index[
        movies["title"] == movie_title
    ][0]

    similarity_scores = list(
        enumerate(
            cosine_similarity(
                genre_matrix[movie_index],
                genre_matrix
            )[0]
        )
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_movies = similarity_scores[1:n+1]

    recommendations = []

    for index, score in top_movies:

        recommendations.append({
            "Movie": movies.iloc[index]["title"],
            "Genres": movies.iloc[index]["genres"],
            "Similarity": round(score, 3)
        })

    return pd.DataFrame(recommendations)


# =========================
# Collaborative Filtering
# =========================

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)


def collaborative_recommendations(user_id, n=5):

    user_index = user_movie_matrix.index.get_loc(user_id)

    similar_users = list(
        enumerate(user_similarity[user_index])
    )

    similar_users = sorted(
        similar_users,
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    scores = {}

    for index, similarity in similar_users:

        for movie_id, rating in user_movie_matrix.iloc[index].items():

            if rating > 0:
                scores[movie_id] = (
                    scores.get(movie_id, 0)
                    + similarity * rating
                )

    watched = user_movie_matrix.loc[user_id]

    scores = {
        movie_id: score
        for movie_id, score in scores.items()
        if watched.get(movie_id, 0) == 0
    }

    top_movies = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n]

    recommendations = []

    for movie_id, score in top_movies:

        title = movies.loc[
            movies["movieId"] == movie_id,
            "title"
        ].values[0]

        recommendations.append(title)

    return recommendations


# =========================
# User Input
# =========================

movie_title = st.selectbox(
    "🎥 Select a movie:",
    movies["title"].tolist()
)

user_id = st.number_input(
    "👤 Enter User ID for personalized recommendations:",
    min_value=1,
    max_value=610,
    value=1,
    step=1
)


# =========================
# Recommendation Button
# =========================

if st.button("🍿 Recommend Movies"):

    # Content-based recommendations
    recommendations = recommend_movies(movie_title)

    st.subheader("🎬 Similar Movies")

    st.dataframe(
        recommendations,
        hide_index=True
    )

    # Collaborative recommendations
    st.subheader("👤 Personalized Recommendations")

    personalized = collaborative_recommendations(
        int(user_id)
    )

    for movie in personalized:
        st.write("🎬", movie)