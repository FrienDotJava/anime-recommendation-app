from helper import load_data, load_params
import tensorflow as tf
from tensorflow import keras
from model import HybridRecommenderNet
import pandas as pd
import numpy as np

def load_train_valid(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_set_path = params['data']['train_set_path']
    valid_set_path = params['data']['valid_set_path']

    train_set = load_data(train_set_path)
    valid_set = load_data(valid_set_path)

    return train_set, valid_set


def split_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[['user', 'anime', 'genre_code']].values
    y = df['rating'].values

    return x, y


def initiate_model(params: dict, merged_df: pd.DataFrame) -> HybridRecommenderNet:
    regularization_strength = params['model']['regularization_strength']
    initializer = params['model']['initializer']
    
    num_users = len(merged_df['user'].unique())
    num_anime = len(merged_df['anime'].unique())
    num_genres = len(merged_df['main_genre'].unique())

    embedding_size = params['model']['embedding_size']
    
    # Use the new HybridRecommenderNet and pass num_genres
    model = HybridRecommenderNet(num_users, num_anime, num_genres, embedding_size, initializer, regularization_strength) 

    return model


def initiate_earlystopping():
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_root_mean_squared_error',
        patience=10,
        min_delta=0.001,
        mode='min',
        restore_best_weights=True
    )


def train_model(params: dict, model: HybridRecommenderNet, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) -> HybridRecommenderNet:
    learning_rate = params['model']['learning_rate']
    epoch = params['model']['epoch']
    batch_size = params['model']['batch_size']

    # model compile
    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    # Model training
    early_stopping = initiate_earlystopping()

    model.fit(
        x = x_train,
        y = y_train,
        batch_size = batch_size,
        epochs = epoch,
        validation_data = (x_valid, y_valid),
        callbacks=[early_stopping],
        verbose=1
    )

    return model


def save_model(model: HybridRecommenderNet, params: dict):
    model_path = params['model']['model_path']
    model.save(model_path)


def main():
    params = load_params()
    merged_data_path = params['data']['merged_data_path']
    
    train_set, valid_set = load_train_valid(params)
    merged_df = load_data(merged_data_path)

    x_train, y_train = split_target(train_set)
    x_valid, y_valid = split_target(valid_set)

    model = initiate_model(params, merged_df)

    model = train_model(params, model, x_train, y_train, x_valid, y_valid)

    save_model(model, params)

if __name__ == "__main__":
    main()