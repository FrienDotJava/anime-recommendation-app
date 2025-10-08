import numpy as np
import pandas as pd
from helper import load_data, load_params, save_data
import os


def load_datasets(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        raw_rating_path = params['data']['raw_rating_path']
        raw_anime_path = params['data']['raw_anime_path']

        rating_df = load_data(raw_rating_path)
        anime_df = load_data(raw_anime_path)
        return rating_df, anime_df
    except Exception as e:
        raise Exception(f"Error loading datasets: {e}")


def clean_rating_data(rating_df: pd.DataFrame) -> pd.DataFrame:
    try:
        rating_df = rating_df.drop_duplicates()
        rating_df.loc[rating_df['rating'] == -1, "rating"] = 0
        rating_df['rating'] = rating_df['rating'].astype(np.float32)
        return rating_df
    except Exception as e:
        raise Exception(f"Error cleaning rating data: {e}")


def preprocess_anime_data(anime_df: pd.DataFrame) -> pd.DataFrame:
    try:
        anime_df['genre'] = anime_df['genre'].fillna('Unknown')
        anime_df['main_genre'] = anime_df['genre'].apply(
            lambda x: x.split(',')[0].strip() if pd.notna(x) and len(x.split(',')) > 0 else 'Unknown'
        )
        return anime_df
    except Exception as e:
        raise Exception(f"Error preprocess anime data: {e}")


def save_datasets(params: dict, rating_cleaned: pd.DataFrame, anime_preprocessed: pd.DataFrame):
    try:
        preprocessed_anime_path = params['data']['preprocessed_anime_path']
        cleaned_rating_path = params['data']['cleaned_rating_path']

        save_data(anime_preprocessed, preprocessed_anime_path)
        save_data(rating_cleaned, cleaned_rating_path)
    except Exception as e:
        raise Exception(f"Error saving dataset: {e}")
    

def main():
    try:
        params = load_params()

        rating_df, anime_df = load_datasets(params)
        rating_df_cleaned = clean_rating_data(rating_df)
        anime_df_preprocessed = preprocess_anime_data(anime_df)

        os.makedirs('data/interim',exist_ok=True)
        save_datasets(params, rating_df_cleaned, anime_df_preprocessed)
    except Exception as e:
        raise Exception(f"Error in main: {e}")


if __name__ == "__main__":
    main()
