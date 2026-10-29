import ssl
import certifi
import streamlit as st
import pandas as pd
import urllib.request
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

DATA_DIR = Path("data")
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


@st.cache_data
def load_data():
    DATA_DIR.mkdir(exist_ok=True)

    if not MOVIES_FILE.exists() or not RATINGS_FILE.exists():
        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            with urllib.request.urlopen(
                DATA_URL,
                timeout=60,
                context=ssl_context
            ) as response:
                archive_bytes = response.read()

            with ZipFile(BytesIO(archive_bytes)) as archive:
                folder = next(
                    (
                        name.split("/")[0]
                        for name in archive.namelist()
                        if name.endswith("/movies.csv")
                    ),
                    None
                )

                if not folder:
                    raise ValueError(
                        "Could not find movies.csv in the dataset."
                    )

                movies_data = archive.read(f"{folder}/movies.csv")
                ratings_data = archive.read(f"{folder}/ratings.csv")

                MOVIES_FILE.write_bytes(movies_data)
                RATINGS_FILE.write_bytes(ratings_data)

        except Exception as exc:
            st.error(
                "Could not download the MovieLens dataset. "
                "Please check your internet connection and SSL certificates."
            )
            st.code(str(exc))
            st.stop()

    movies = pd.read_csv(MOVIES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)

    movies["genres"] = (
        movies["genres"]
        .fillna("")
        .replace("(no genres listed)", "")
    )
    movies["title"] = movies["title"].fillna("Unknown title")

    return movies, ratings


@st.cache_data
def build_similarity(movies):
    genre_text = (
        movies["genres"]
        .fillna("")
        .str.replace("|", " ", regex=False)
    )

    vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
    matrix = vectorizer.fit_transform(genre_text)

    return cosine_similarity(matrix)


movies, ratings = load_data()
similarity = build_similarity(movies)

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}
.hero {
    padding: 1.5rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #20123a, #172554, #111827);
    margin-bottom: 1rem;
}
.hero h1 {
    margin: 0;
    color: white;
    font-size: 2.5rem;
}
.hero p {
    color: #d1d5db;
    margin-bottom: 0;
}
.movie-card {
    padding: 1rem;
    border: 1px solid #303846;
    border-radius: 12px;
    background: #171b25;
    margin-bottom: 1rem;
}
.movie-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f9fafb;
}
.movie-meta {
    color: #aab4c5;
    font-size: .88rem;
    margin-top: .4rem;
}
</style>

<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Your next favourite movie is just one recommendation away.</p>
</div>
""", unsafe_allow_html=True)

total_movies = len(movies)
total_ratings = len(ratings)
total_users = ratings["userId"].nunique()

m1, m2, m3 = st.columns(3)
m1.metric("Movies in catalogue", f"{total_movies:,}")
m2.metric("User ratings", f"{total_ratings:,}")
m3.metric("MovieLens users", f"{total_users:,}")

st.divider()

tab1, tab2 = st.tabs([
    "✨ Similar Movies",
    "⭐ Popular Movies"
])

with tab1:
    st.subheader("Find movies similar to your favourite")

    movie_titles = movies["title"].tolist()

    default_index = (
        movie_titles.index("Toy Story (1995)")
        if "Toy Story (1995)" in movie_titles
        else 0
    )

    selected_title = st.selectbox(
        "Choose a movie",
        movie_titles,
        index=default_index
    )

    count = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    if st.button(
        "Recommend Movies",
        type="primary",
        use_container_width=True
    ):
        selected_index = movies.index[
            movies["title"] == selected_title
        ][0]

        scores = list(enumerate(similarity[selected_index]))
        scores.sort(key=lambda item: item[1], reverse=True)

        recommendations = [
            (idx, score)
            for idx, score in scores
            if idx != selected_index
        ][:count]

        st.success(
            f"Movies similar to **{selected_title}**:"
        )

        cols = st.columns(2)

        for i, (idx, score) in enumerate(recommendations):
            row = movies.iloc[idx]

            with cols[i % 2]:
                genres = row["genres"].replace("|", ", ")

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">
                            🎞️ {row["title"]}
                        </div>
                        <div class="movie-meta">
                            Genres: {genres or "Not specified"}
                        </div>
                        <div class="movie-meta">
                            Genre similarity: {score:.0%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.caption(
        "Method: Content-based filtering using movie genres "
        "and cosine similarity."
    )


with tab2:
    st.subheader("Popular movies based on community ratings")

    min_ratings = st.slider(
        "Minimum number of ratings",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )

    movie_stats = (
        ratings.groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    popular = movies.merge(
        movie_stats,
        on="movieId",
        how="left"
    )

    popular = (
        popular[popular["rating_count"] >= min_ratings]
        .sort_values(
            ["avg_rating", "rating_count"],
            ascending=False
        )
        .head(20)
        .copy()
    )

    if popular.empty:
        st.info(
            "No movies meet this minimum. "
            "Try lowering the slider."
        )
    else:
        popular["avg_rating"] = popular["avg_rating"].round(2)

        st.dataframe(
            popular[
                ["title", "genres", "avg_rating", "rating_count"]
            ].rename(
                columns={
                    "title": "Movie",
                    "genres": "Genres",
                    "avg_rating": "Average rating",
                    "rating_count": "Number of ratings"
                }
            ),
            use_container_width=True,
            hide_index=True
        )


with st.sidebar:
    st.header("About CineMatch")
    st.write(
        "CineMatch recommends movies by comparing their genres."
    )
    st.write(
        "**Algorithm:** CountVectorizer + Cosine Similarity"
    )
    st.write("**Dataset:** MovieLens Latest Small")
    st.markdown(
        "[MovieLens dataset information]"
        "(https://grouplens.org/datasets/movielens/)"
    )
    st.caption(
        "The dataset downloads automatically on the first run."
    )
