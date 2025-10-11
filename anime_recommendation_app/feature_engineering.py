import pandas as pd
import os
from helper import load_data, load_params, save_params, save_json, save_data

def load_datasets(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        preprocessed_anime_path = params['data']['preprocessed_anime_path']
        cleaned_rating_path = params['data']['cleaned_rating_path']

        anime_df = load_data(preprocessed_anime_path)
        rating_df = load_data(cleaned_rating_path)
        return rating_df, anime_df
    except Exception as e:
        raise Exception(f"Error loading datasets: {e}")


def merge_datasets(rating_df: pd.DataFrame, anime_df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Merge datasets to get 'type' and 'main_genre' columns
        df_merge = pd.merge(rating_df, anime_df[['anime_id', 'type', 'main_genre']], on='anime_id')
        return df_merge
    except Exception as e:
        raise Exception(f"Error mergin datasets: {e}")


def filter_user(df: pd.DataFrame, MIN_USER_RATINGS: int) -> pd.DataFrame:
    try:
        user_counts = df['user_id'].value_counts()
        active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
        return df[df['user_id'].isin(active_users)]
    except Exception as e:
        raise Exception(f"Error filter user: {e}")


def filter_anime(df: pd.DataFrame, MIN_ANIME_RATINGS: int) -> pd.DataFrame:
    try:
        anime_counts = df['anime_id'].value_counts()
        popular_anime = anime_counts[anime_counts >= MIN_ANIME_RATINGS].index
        return df[df['anime_id'].isin(popular_anime)]
    except Exception as e:
        raise Exception(f"Error filtering anime: {e}")

def filter_dataset(params: dict, df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df[df['type'] == 'TV'].copy()

        MIN_USER_RATINGS = params['data']['min_user_ratings']
        MIN_ANIME_RATINGS = params['data']['min_anime_ratings']

        print(f"Initial filtered TV series ratings: {len(df)}.")

        df = filter_user(df, MIN_USER_RATINGS)

        df = filter_anime(df, MIN_ANIME_RATINGS)
        print(f"Final data size after filtering (100/100): {len(df)} ratings.")
        return df
    except Exception as e:
        raise Exception(f"Error filtering dataset: {e}")


def encode_user(params: dict, df: pd.DataFrame) -> dict:
    try:
        user_ids = df['user_id'].unique().tolist()
        user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}

        save_path = params['feature_engineering']['user_to_user_encoded_path']
        save_json(user_to_user_encoded, save_path)

        return user_to_user_encoded
    except Exception as e:
        raise Exception(f"Error encoding user: {e}")


def encode_anime(params: dict, df: pd.DataFrame) -> dict:
    try:
        anime_ids = df['anime_id'].unique().tolist()
        anime_to_anime_encoded = {x: i for i, x in enumerate(anime_ids)}
        anime_encoded_to_anime = {i: x for i, x in enumerate(anime_ids)}

        save_path = params['feature_engineering']['anime_encoded_to_anime_path']
        save_path2 = params['feature_engineering']['anime_to_anime_encoded_path']
        save_json(anime_encoded_to_anime, save_path)
        save_json(anime_to_anime_encoded, save_path2)

        return anime_to_anime_encoded
    except Exception as e:
        raise Exception(f"Error encoding anime: {e}")


def encode_genre(params: dict, df: pd.DataFrame) -> dict:
    try:
        genre_names = df['main_genre'].unique().tolist()
        genre_to_genre_encoded = {x: i for i, x in enumerate(genre_names)}

        save_path = params['feature_engineering']['genre_to_genre_encoded_path']
        save_json(genre_to_genre_encoded, save_path)

        return genre_to_genre_encoded
    except Exception as e:
        raise Exception(f"Error encoding genre: {e}")


def encode_dataset(params: dict, df: pd.DataFrame) -> pd.DataFrame:
    try:
        os.makedirs('models/artifacts', exist_ok=True)

        user_to_user_encoded = encode_user(params, df)
        anime_to_anime_encoded = encode_anime(params, df)
        genre_to_genre_encoded = encode_genre(params, df)
        
        # Mapping encoded features to the DataFrame
        df['user'] = df['user_id'].map(user_to_user_encoded)
        df['anime'] = df['anime_id'].map(anime_to_anime_encoded)
        df['genre_code'] = df['main_genre'].map(genre_to_genre_encoded)

        return df
    except Exception as e:
        raise Exception(f"Error encoding dataset: {e}")


def scale_rating(df: pd.DataFrame, min_rating: int, max_rating: int) -> pd.DataFrame:
    try:
        df['rating'] = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))
        return df
    except Exception as e:
        raise Exception(f"Error scaling rating: {e}")


def shuffle_dataset(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.sample(frac=1, random_state=42)
    except Exception as e:
        raise Exception(f"Error shuffling dataset: {e}")


def main():
    try:
        params = load_params()
        rating_df_cleaned, anime_df_preprocessed = load_datasets(params)

        rating_df_merged = merge_datasets(rating_df_cleaned, anime_df_preprocessed)

        
        rating_df_cleaned = filter_dataset(params, rating_df_merged)

        min_rating = rating_df_cleaned['rating'].min()
        max_rating = rating_df_cleaned['rating'].max()

        rating_df_cleaned = encode_dataset(params, rating_df_cleaned)

        rating_df_cleaned = scale_rating(rating_df_cleaned, min_rating, max_rating)

        rating_df_cleaned = shuffle_dataset(rating_df_cleaned)

        save_data(rating_df_cleaned,params['data']['merged_data_path'])
        save_params(params)
        save_json({"min": float(min_rating), "max": float(max_rating)}, params['feature_engineering']['rating_scale_path'])
    except Exception as e:
        raise Exception(f"Error in main: {e}")


if __name__ == "__main__":
    main()