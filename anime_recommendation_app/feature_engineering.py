import pandas as pd
import os
from helper import load_data, load_params, save_params, save_json, save_data

def load_datasets(params):
    preprocessed_anime_path = params['data']['preprocessed_anime_path']
    cleaned_rating_path = params['data']['cleaned_rating_path']

    anime_df = load_data(preprocessed_anime_path)
    rating_df = load_data(cleaned_rating_path)
    return rating_df, anime_df


def merge_datasets(rating_df, anime_df):
    # Merge datasets to get 'type' and 'main_genre' columns
    df_merge = pd.merge(rating_df, anime_df[['anime_id', 'type', 'main_genre']], on='anime_id')
    return df_merge


def filter_user(df, MIN_USER_RATINGS):
    user_counts = df['user_id'].value_counts()
    active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    return df[df['user_id'].isin(active_users)]


def filter_anime(df, MIN_ANIME_RATINGS):
    anime_counts = df['anime_id'].value_counts()
    popular_anime = anime_counts[anime_counts >= MIN_ANIME_RATINGS].index
    return df[df['anime_id'].isin(popular_anime)]


def filter_dataset(params, df):
    df = df[df['type'] == 'TV'].copy()

    MIN_USER_RATINGS = params['data']['min_user_ratings']
    MIN_ANIME_RATINGS = params['data']['min_anime_ratings']

    print(f"Initial filtered TV series ratings: {len(df)}.")

    df = filter_user(df, MIN_USER_RATINGS)

    df = filter_anime(df, MIN_ANIME_RATINGS)
    print(f"Final data size after filtering (100/100): {len(df)} ratings.")
    return df


def encode_user(df):
    user_ids = df['user_id'].unique().tolist()
    return {x: i for i, x in enumerate(user_ids)}


def encode_anime(params, df):
    anime_ids = df['anime_id'].unique().tolist()
    anime_to_anime_encoded = {x: i for i, x in enumerate(anime_ids)}
    anime_encoded_to_anime = {i: x for i, x in enumerate(anime_ids)}

    os.makedirs('models/artifacts', exist_ok=True)
    save_path = params['feature_engineering']['anime_encoded_to_anime_path']
    save_json(anime_encoded_to_anime, save_path)

    return anime_to_anime_encoded


def encode_genre(df):
    genre_names = df['main_genre'].unique().tolist()
    return {x: i for i, x in enumerate(genre_names)}


def encode_dataset(params, df):
    # --- ENCODING ALL FEATURES (User, Anime, Genre) ---

    # Encoding User and Anime IDs
    user_to_user_encoded = encode_user(df)

    anime_to_anime_encoded = encode_anime(params, df)

    # Encoding Main Genre
    genre_to_genre_encoded = encode_genre(df)

    # Mapping encoded features to the DataFrame
    df['user'] = df['user_id'].map(user_to_user_encoded)
    df['anime'] = df['anime_id'].map(anime_to_anime_encoded)
    df['genre_code'] = df['main_genre'].map(genre_to_genre_encoded)

    return df


def scale_rating(df, min_rating, max_rating):
    df['rating'] = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))
    return df


def shuffle_dataset(df):
    return df.sample(frac=1, random_state=42)


def main():
    params = load_params()
    rating_df_cleaned, anime_df_preprocessed = load_datasets(params)

    rating_df_merged = merge_datasets(rating_df_cleaned, anime_df_preprocessed)

    
    rating_df_cleaned = filter_dataset(rating_df_merged)

    min_rating = rating_df_cleaned['rating'].min()
    max_rating = rating_df_cleaned['rating'].max()

    rating_df_cleaned = encode_dataset(params, rating_df_cleaned)

    rating_df_cleaned = scale_rating(rating_df_cleaned, min_rating, max_rating)

    rating_df_cleaned = shuffle_dataset(rating_df_cleaned)

    save_data(rating_df_cleaned,params['data']['merged_data_path'])
    save_params(params)


if __name__ == "__main__":
    main()