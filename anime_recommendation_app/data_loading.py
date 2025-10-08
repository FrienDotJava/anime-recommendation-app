import kagglehub
from kagglehub import KaggleDatasetAdapter
from helper import load_params,save_data
import pandas as pd

def load_from_kaggle(filename: str) -> pd.DataFrame:
    return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "CooperUnion/anime-recommendations-database", filename)

def main():
    params = load_params()
    raw_anime_path = params['data']['raw_anime_path']
    raw_rating_path = params['data']['raw_rating_path']

    anime_df = load_from_kaggle("anime.csv")

    rating_df = load_from_kaggle("rating.csv")

    save_data(anime_df,raw_anime_path)
    save_data(rating_df,raw_rating_path)

if __file__ == "__main__":
    main()