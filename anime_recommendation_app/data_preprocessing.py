from helper import load_data, load_params, save_data
import pandas as pd

def split_data(df: pd.DataFrame, valid_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_indices = int(valid_size * df.shape[0])
        train_set, test_set = (
            df[:train_indices],
            df[train_indices:]
        )
        return train_set, test_set
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")


def main():
    try:
        params = load_params()

        merged_data_path = params['data']['merged_data_path']
        train_set_path = params['data']['train_set_path']
        test_set_path = params['data']['test_set_path']
        VALID_SIZE = params['data_preprocessing']['valid_size']

        merged_df = load_data(merged_data_path)

        train_set, test_set = split_data(merged_df, VALID_SIZE)

        save_data(train_set, train_set_path)
        save_data(test_set, test_set_path)
    except Exception as e:
        raise Exception(f"Error in main: {e}")


if __name__ == "__main__":
    main()