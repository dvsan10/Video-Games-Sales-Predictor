from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Video Game Sales Predictor",
    version="1.0"
)

ARTIFACT_PATH = Path(__file__).with_name(
    "video_game_sales_model.joblib"
)

artifacts = joblib.load(ARTIFACT_PATH)

# Final Polynomial Regression model
model = artifacts["model"]
poly = artifacts["poly"]

# Preprocessing
selector = artifacts["selector"]
scaler = artifacts["scaler"]
genre_encoder = artifacts["genre_encoder"]
console_frequency = artifacts["console_frequency"]

# Historical features
publisher_history = artifacts["publisher_history"]
developer_history = artifacts["developer_history"]

# Missing-value handling
critic_median = artifacts["critic_median"]


class GameInput(BaseModel):
    title: str
    console: str
    genre: str
    publisher: str
    developer: str
    critic_score: float | None = None


def build_features(game: GameInput):

    title = game.title or ""
    publisher = game.publisher or "Unknown"
    developer = game.developer or "Unknown"
    console = game.console or "Unknown"
    genre = game.genre or "Unknown"

    publisher_games = publisher_history["previous_games"].get(
        publisher, 0
    )

    publisher_sales = publisher_history["previous_sales"].get(
        publisher, 0
    )

    developer_games = developer_history["previous_games"].get(
        developer, 0
    )

    developer_sales = developer_history["previous_sales"].get(
        developer, 0
    )

    row = pd.DataFrame([{
        "critic_score":
            critic_median
            if game.critic_score is None
            else game.critic_score,

        "publisher_avg_sales":
            publisher_sales / publisher_games
            if publisher_games else 0,

        "publisher_previous_games":
            publisher_games,

        "developer_avg_sales":
            developer_sales / developer_games
            if developer_games else 0,

        "developer_previous_games":
            developer_games,

        "title_length":
            len(title),

        "title_word_count":
            len(title.split()),

        "console_frequency":
            console_frequency.get(console, 0)
    }])

    genre_encoded = genre_encoder.transform(
        pd.DataFrame({"genre": [genre]})
    )

    genre_df = pd.DataFrame(
        genre_encoded,
        columns=genre_encoder.get_feature_names_out(["genre"])
    )

    features = pd.concat(
        [row, genre_df],
        axis=1
    )

    features = features.reindex(
        columns=selector.feature_names_in_,
        fill_value=0
    )

    selected = selector.transform(features)

    scaled = scaler.transform(selected)

    # Polynomial transformation used during model training
    polynomial_features = poly.transform(scaled)

    return polynomial_features


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(game: GameInput):

    X = build_features(game)

    log_prediction = model.predict(X)[0]

    sales_prediction = float(
        np.expm1(log_prediction)
    )

    return {
        "predicted_total_sales":
            round(max(sales_prediction, 0.0), 4)
    }