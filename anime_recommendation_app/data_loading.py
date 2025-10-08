import kagglehub
from kagglehub import KaggleDatasetAdapter
from helper import load_params,save_data
import pandas as pd
import os


def load_from_kaggle(filename: str) -> pd.DataFrame:
    try:
        return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "CooperUnion/anime-recommendations-database", filename)
    except Exception as e:
        raise Exception(f"Error load from kaggle: {e}")

def main():
    try:
        params = load_params()
        raw_anime_path = params['data']['raw_anime_path']
        raw_rating_path = params['data']['raw_rating_path']

        anime_df = load_from_kaggle("anime.csv")

        rating_df = load_from_kaggle("rating.csv")
        print("print")

        raw_path = os.path.join('data','raw')
        os.makedirs(raw_path)

        save_data(anime_df,raw_anime_path)
        save_data(rating_df,raw_rating_path)
    except Exception as e:
        raise Exception(f"Error in main: {e}")

if __name__ == "__main__":
    main()