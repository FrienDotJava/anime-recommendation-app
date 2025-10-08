import numpy as np
import pandas as pd
from helper import load_data, load_params, save_data
import os


def load_datasets(params):
    raw_rating_path = params['data']['raw_rating_path']
    raw_anime_path = params['data']['raw_anime_path']

    rating_df = load_data(raw_rating_path)
    anime_df = load_data(raw_anime_path)
    return rating_df, anime_df


def clean_rating_data(rating_df):
    rating_df = rating_df.drop_duplicates()
    rating_df.loc[rating_df['rating'] == -1, "rating"] = 0
    rating_df['rating'] = rating_df['rating'].astype(np.float32)
    return rating_df


def preprocess_anime_data(anime_df):
    anime_df['genre'] = anime_df['genre'].fillna('Unknown')
    anime_df['main_genre'] = anime_df['genre'].apply(
        lambda x: x.split(',')[0].strip() if pd.notna(x) and len(x.split(',')) > 0 else 'Unknown'
    )
    return anime_df


def save_datasets(params, rating_cleaned, anime_preprocessed):
    preprocessed_anime_path = params['data']['preprocessed_anime_path']
    cleaned_rating_path = params['data']['cleaned_rating_path']

    save_data(anime_preprocessed, preprocessed_anime_path)
    save_data(rating_cleaned, cleaned_rating_path)


def main():
    params = load_params()

    rating_df, anime_df = load_datasets(params)
    rating_df_cleaned = clean_rating_data(rating_df)
    anime_df_preprocessed = preprocess_anime_data(anime_df)

    os.makedirs('data/interim',exist_ok=True)
    save_datasets(params, rating_df_cleaned, anime_df_preprocessed)


if __name__ == "__main__":
    main()
