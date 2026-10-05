# 🎬 Movie Recommendation System

A machine learning-based movie recommendation system that recommends movies based on genres and user rating patterns.

## Features

- Content-based movie recommendations
- Collaborative filtering
- Personalized recommendations using User ID
- Top 5 movie recommendations
- Interactive Streamlit web interface
- Movie genre similarity using TF-IDF

## How It Works

### Content-Based Filtering

The system uses movie genres to find similar movies.

TF-IDF converts movie genres into numerical features, and cosine similarity is used to find movies with similar genres.

### Collaborative Filtering

The system uses MovieLens user-rating data to identify users with similar rating patterns.

Recommendations are generated based on movies rated by similar users.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Dataset

This project uses the MovieLens Small Dataset provided by GroupLens Research.

Dataset: https://grouplens.org/datasets/movielens/

## Project Structure

Movie-Recommendation-System/
│
├── app.py
├── movie_recommendation.ipynb
├── requirements.txt
├── README.md
│
└── data/
    └── ml-latest-small/
        ├── movies.csv
        ├── ratings.csv
        ├── tags.csv
        ├── links.csv
        └── README.txt

## How to Run

Install the required libraries:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

The application will open in your browser.

## Output

The system provides:

- Similar movies based on genres
- Personalized movie recommendations based on user ratings

## Project

Machine Learning Internship Project