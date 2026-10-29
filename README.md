# CineMatch — Movie Recommendation System

A Python + Streamlit movie recommender built with the MovieLens dataset. It provides content-based recommendations and a popular-movies list.

## Features
- Choose a movie and receive the top 5–20 similar movies.
- Compare movies by genre using cosine similarity.
- View popular movies based on average community ratings and a minimum rating count.
- Interactive Streamlit interface.
- Automatically downloads the MovieLens Latest Small dataset on the first run.

## Requirements
- Python 3.10 or newer
- Internet connection for the first run (to download MovieLens data)

## Setup (Windows PowerShell)

```powershell
mkdir MovieRecommendationSystem
cd MovieRecommendationSystem
# Extract the downloaded project ZIP into this folder first
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this in the same terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open the local URL Streamlit displays, usually `http://localhost:8501`.

## How it works
1. Reads movie titles and genres from `movies.csv`.
2. Converts genres into numeric vectors using `CountVectorizer`.
3. Calculates cosine similarity between movie vectors.
4. Ranks movies by similarity to the selected title.
5. Uses ratings data to show popular movies.

## Dataset
Uses the official [MovieLens Latest Small dataset](https://grouplens.org/datasets/movielens/), containing about 9,700 movies and 100,000 ratings. The dataset is downloaded automatically and stored under `data/`.

## Project structure
```text
MovieRecommendationSystem/
├── app.py
├── requirements.txt
├── README.md
└── data/                 # created automatically on first run
    ├── movies.csv
    └── ratings.csv
```

## Limitations
Recommendations are based on genre similarity, not personal user profiles. Popular-movie ranking is based on ratings in the supplied dataset. This is an educational project; review the dataset terms before redistributing data.
