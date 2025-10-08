from helper import load_data, load_params, save_data


def split_data(df, valid_size):
    train_indices = int(valid_size * df.shape[0])
    train_set, test_set = (
        df[:train_indices],
        df[train_indices:]
    )
    return train_set, test_set


def main():
    params = load_params()

    merged_data_path = params['data']['merged_data_path']
    train_set_path = params['data']['train_set_path']
    test_set_path = params['data']['test_set_path']
    VALID_SIZE = params['data_preprocessing']['valid_size']

    merged_df = load_data(merged_data_path)

    train_set, test_set = split_data(merged_df, VALID_SIZE)

    save_data(train_set, train_set_path)
    save_data(test_set, test_set_path)


if __name__ == "__main__":
    main()