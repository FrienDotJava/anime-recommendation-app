import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from anime_recommendation_app.modeling.model import HybridRecommenderNet

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./models/artifacts")
MODEL_PATH   = os.getenv("MODEL_PATH",   "./models/model.keras")
ANIME_CSV    = os.getenv("ANIME_CSV",    "./data/raw/anime.csv")

USER_TO_ENC_PATH   = os.path.join(ARTIFACT_DIR, "user_to_user_encoded.json")
ANIME_TO_ENC_PATH  = os.path.join(ARTIFACT_DIR, "anime_to_anime_encoded.json")
GENRE_TO_ENC_PATH  = os.path.join(ARTIFACT_DIR, "genre_to_genre_encoded.json")
ANIME_ENC_TO_ID    = os.path.join(ARTIFACT_DIR, "anime_encoded_to_anime.json")
SCALE_PATH         = os.path.join(ARTIFACT_DIR, "rating_scale.json")

app = FastAPI(title="Anime Hybrid Recommender API", version="1.0.0")  

class PredictRequest(BaseModel):
    user_id: int
    anime_id: int

class PredictResponse(BaseModel):
    user_id: int
    anime_id: int
    predicted_score_0_1: float = Field(..., description="Model output in [0,1]")
    predicted_rating: Optional[float] = Field(None, description="Denormalized rating (e.g., 0–10)")

class RecommendRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="Known user. If None, use cold-start via preferred_genres.")
    top_k: int = 10
    allowed_genres: Optional[List[str]] = None
    exclude_anime_ids: Optional[List[int]] = None
    only_type: Optional[str] = Field(None, description="e.g., 'TV', 'Movie'")

    preferred_genres: Optional[List[str]] = None

class RecommendedItem(BaseModel):
    anime_id: int
    name: Optional[str]
    main_genre: Optional[str]
    predicted_score_0_1: float

class RecommendResponse(BaseModel):
    items: List[RecommendedItem]

@app.on_event("startup")
def load_artifacts():
    global model, anime_df, user_to_enc, anime_to_enc, genre_to_enc, enc_to_anime, rating_scale

    # Load model
    model = tf.keras.models.load_model(
                MODEL_PATH, 
                custom_objects={'HybridRecommenderNet': HybridRecommenderNet}
            )

    # Load anime_df and create main_genre column
    anime_df = pd.read_csv(ANIME_CSV)
    if "genre" not in anime_df.columns:
        anime_df["genre"] = "Unknown"
    anime_df["genre"] = anime_df["genre"].fillna("Unknown")
    anime_df["main_genre"] = anime_df["genre"].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) and x else "Unknown"
    )

    # Load encoders
    with open(USER_TO_ENC_PATH, "r") as f:
        user_to_enc = {int(k): int(v) for k, v in json.load(f).items()}
    with open(ANIME_TO_ENC_PATH, "r") as f:
        anime_to_enc = {int(k): int(v) for k, v in json.load(f).items()}
    with open(GENRE_TO_ENC_PATH, "r") as f:
        genre_to_enc = json.load(f)

    enc_to_anime = None
    if os.path.exists(ANIME_ENC_TO_ID):
        with open(ANIME_ENC_TO_ID, "r") as f:
            enc_to_anime = {int(k): int(v) for k, v in json.load(f).items()}

    rating_scale = {"min": 0.0, "max": 10.0}
    if os.path.exists(SCALE_PATH):
        with open(SCALE_PATH, "r") as f:
            rating_scale = json.load(f)
            
    global COLD_BASE_INDEX
    max_known = max(user_to_enc.values()) if len(user_to_enc) > 0 else -1
    COLD_BASE_INDEX = max_known + 1


def encode_row(user_id: int, anime_id: int) -> np.ndarray:
    if user_id not in user_to_enc:
        raise KeyError("unknown_user")
    if anime_id not in anime_to_enc:
        raise KeyError("unknown_anime")

    row = anime_df.loc[anime_df["anime_id"] == anime_id]
    if row.empty:
        raise KeyError("anime_not_found_in_master")
    main_genre = row.iloc[0]["main_genre"]
    if main_genre not in genre_to_enc:
        raise KeyError("unknown_genre")

    user_code = user_to_enc[user_id]
    anime_code = anime_to_enc[anime_id]
    genre_code = genre_to_enc[main_genre]
    return np.array([[user_code, anime_code, genre_code]], dtype=np.int64)


def denormalize(y_pred: float) -> float:
    return rating_scale["min"] + y_pred * (rating_scale["max"] - rating_scale["min"])


def filter_candidate_anime(
    allowed_genres: Optional[List[str]], only_type: Optional[str], exclude_anime_ids: Optional[List[int]]
) -> pd.DataFrame:
    df = anime_df
    if only_type:
        df = df[df["type"] == only_type]
    if allowed_genres and len(allowed_genres) > 0:
        df = df[df["main_genre"].isin(allowed_genres)]
    if exclude_anime_ids and len(exclude_anime_ids) > 0:
        df = df[~df["anime_id"].isin(exclude_anime_ids)]
    
    df = df[df["anime_id"].isin(anime_to_enc.keys())]
    
    df = df[df["main_genre"].isin(genre_to_enc.keys())]
    return df.copy()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X = encode_row(req.user_id, req.anime_id)
    except KeyError as e:
        msg = str(e)
        if "unknown_user" in msg:
            raise HTTPException(status_code=400, detail="User not found in trained encoders.")
        if "unknown_anime" in msg or "anime_not_found_in_master" in msg:
            raise HTTPException(status_code=400, detail="Anime not found or not in trained encoders.")
        if "unknown_genre" in msg:
            raise HTTPException(status_code=400, detail="Anime main_genre not recognized by encoder.")
        raise

    y_pred = float(model.predict(X, verbose=0).reshape(-1)[0])
    out = PredictResponse(
        user_id=req.user_id,
        anime_id=req.anime_id,
        predicted_score_0_1=y_pred,
        predicted_rating=round(denormalize(y_pred), 3)
    )
    return out

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    candidates = filter_candidate_anime(req.allowed_genres, req.only_type, req.exclude_anime_ids)

    # If we have a known user, give recommendation based on preference
    if req.user_id is not None and req.user_id in user_to_enc:
        user_code = user_to_enc[req.user_id]
        # Build [user_code, anime_code, genre_code] for all candidates
        anime_codes = candidates["anime_id"].map(anime_to_enc)
        genre_codes = candidates["main_genre"].map(genre_to_enc)
        X = np.column_stack([np.full(len(candidates), user_code, dtype=np.int64),
                             anime_codes.values.astype(np.int64),
                             genre_codes.values.astype(np.int64)])
        y_pred = model.predict(X, verbose=0).reshape(-1)
        candidates = candidates.assign(score=y_pred)
        top = candidates.sort_values("score", ascending=False).head(req.top_k)

    else:
        # Cold-start: score by genre preference if provided. Otherwise, return popular/random
        if not req.preferred_genres:
            # simple neutral score = 0.5
            candidates = candidates.assign(score=0.5)
            top = candidates.sample(n=min(req.top_k, len(candidates)), random_state=42)
        else:
            # simple heuristic: preferred genre gets 0.7, others 0.4
            candidates = candidates.assign(
                score=np.where(candidates["main_genre"].isin(req.preferred_genres), 0.7, 0.4)
            )
            top = candidates.sort_values("score", ascending=False).head(req.top_k)

    items = [
        RecommendedItem(
            anime_id=int(r.anime_id),
            name=r.get("name") if "name" in r.index else None,
            main_genre=r.get("main_genre") if "main_genre" in r.index else None,
            predicted_score_0_1=float(r.score)
        )
        for _, r in top.iterrows()
    ]
    return RecommendResponse(items=items)

COLD_SLOTS = int(os.getenv("COLD_SLOTS", "1000"))
COLD_BASE_INDEX = None  # set at startup based on loaded encoders
cold_slot_in_use: Dict[str, int] = {}  # map session/user token

class RatedItem(BaseModel):
    anime_id: int
    rating: float 

class BootstrapRequest(BaseModel):
    session_key: str = Field(..., description="Your client session/user token")
    rated: List[RatedItem]
    top_k: int = 10
    allowed_genres: Optional[List[str]] = None
    only_type: Optional[str] = None

class BootstrapResponse(BaseModel):
    personalized_user_code: int
    items: List[RecommendedItem]

def get_or_assign_cold_slot(session_key: str) -> int:
    # reuse if already assigned this session
    if session_key in cold_slot_in_use:
        return cold_slot_in_use[session_key]
    # find next free slot
    for i in range(COLD_SLOTS):
        slot = COLD_BASE_INDEX + i
        if slot not in cold_slot_in_use.values():
            cold_slot_in_use[session_key] = slot
            return slot
    raise HTTPException(status_code=429, detail="No cold-start slots available right now.")



def _normalize(y: np.ndarray) -> np.ndarray:
    return (y - rating_scale["min"]) / max(1e-8, (rating_scale["max"] - rating_scale["min"]))

def _prepare_bootstrap_xy(rated: List[RatedItem], cold_user_code: int):
    xs, ys = [], []
    for r in rated:
        if r.anime_id not in anime_to_enc:
            # skip unknown items to the model
            continue
        row = anime_df.loc[anime_df["anime_id"] == r.anime_id]
        if row.empty:
            continue
        main_genre = row.iloc[0]["main_genre"]
        if main_genre not in genre_to_enc:
            continue
        xs.append([cold_user_code, anime_to_enc[r.anime_id], genre_to_enc[main_genre]])
        ys.append(r.rating)
    if not xs:
        raise HTTPException(status_code=400, detail="None of the provided anime exist in the trained encoders.")
    X = np.array(xs, dtype=np.int64)
    y = _normalize(np.array(ys, dtype=np.float32))
    return X, y

@app.post("/bootstrap_recommend", response_model=BootstrapResponse)
@app.post("/bootstrap_recommend", response_model=BootstrapResponse)
def bootstrap_recommend(req: BootstrapRequest):
    # Pick or create a cold slot
    cold_user_code = get_or_assign_cold_slot(req.session_key)

    # Build training mini-batch from user’s rated items
    X, y = _prepare_bootstrap_xy(req.rated, cold_user_code)

    for layer in model.layers:
        layer.trainable = False
    try:
        model.user_embedding.trainable = True
        model.user_bias.trainable = True
    except Exception:
        for l in model.layers:
            if "user_embedding" in l.name or "user_bias" in l.name:
                l.trainable = True

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    model.fit(X, y, batch_size=min(64, len(X)), epochs=5, verbose=0)

    # Build candidate pool and EXCLUDE already-rated items
    rated_ids = {int(r.anime_id) for r in req.rated}
    candidates = filter_candidate_anime(
        req.allowed_genres,
        req.only_type,
        exclude_anime_ids=list(rated_ids),
    )
    if candidates.empty:
        raise HTTPException(status_code=404, detail="No candidates after filters.")

    # Score
    anime_codes = candidates["anime_id"].map(anime_to_enc)
    genre_codes = candidates["main_genre"].map(genre_to_enc)
    Xc = np.column_stack([
        np.full(len(candidates), cold_user_code, dtype=np.int64),
        anime_codes.values.astype(np.int64),
        genre_codes.values.astype(np.int64),
    ])
    y_pred = model.predict(Xc, verbose=0).reshape(-1)
    candidates = candidates.assign(score=y_pred)
    top = candidates.sort_values("score", ascending=False).head(req.top_k)

    items = [
        RecommendedItem(
            anime_id=int(r.anime_id),
            name=r.get("name") if "name" in r.index else None,
            main_genre=r.get("main_genre") if "main_genre" in r.index else None,
            predicted_score_0_1=float(r.score),
        )
        for _, r in top.iterrows()
    ]
    return BootstrapResponse(personalized_user_code=cold_user_code, items=items)
