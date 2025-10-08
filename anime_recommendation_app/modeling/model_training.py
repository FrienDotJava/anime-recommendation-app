from anime_recommendation_app.helper import load_data, load_params
import tensorflow as tf
from tensorflow import keras
from .model import HybridRecommenderNet
import pandas as pd
import numpy as np

def load_train_test(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_set_path = params['data']['train_set_path']
        test_set_path = params['data']['test_set_path']

        train_set = load_data(train_set_path)
        test_set = load_data(test_set_path)

        return train_set, test_set
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def split_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = df[['user', 'anime', 'genre_code']].values
        y = df['rating'].values

        return x, y
    except Exception as e:
        raise Exception(f"Error splitting target: {e}")


def initiate_model(params: dict, merged_df: pd.DataFrame) -> HybridRecommenderNet:
    try:
        regularization_strength = float(params['model']['regularization_strength'])
        initializer = params['model']['initializer']
        
        num_users = len(merged_df['user'].unique())
        num_anime = len(merged_df['anime'].unique())
        num_genres = len(merged_df['main_genre'].unique())

        embedding_size = params['model']['embedding_size']
        
        # Use the new HybridRecommenderNet and pass num_genres
        model = HybridRecommenderNet(num_users, num_anime, num_genres, embedding_size, initializer, regularization_strength) 

        return model
    except Exception as e:
        raise Exception(f"Error initiating model: {e}")


def initiate_earlystopping():
    try:
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_root_mean_squared_error',
            patience=10,
            min_delta=0.001,
            mode='min',
            restore_best_weights=True
        )
    except Exception as e:
        raise Exception(f"Error initiating earlystopping: {e}")


def train_model(params: dict, model: HybridRecommenderNet, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> HybridRecommenderNet:
    try:
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
            validation_data = (x_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )

        return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")


def save_model(model: HybridRecommenderNet, params: dict):
    try:
        model_path = params['model']['model_path']
        model.save(model_path)
    except Exception as e:
        raise Exception(f"Error saving model: {e}")


def main():
    try:
        params = load_params()
        merged_data_path = params['data']['merged_data_path']
        
        train_set, test_set = load_train_test(params)
        merged_df = load_data(merged_data_path)

        x_train, y_train = split_target(train_set)
        x_test, y_test = split_target(test_set)

        model = initiate_model(params, merged_df)

        model = train_model(params, model, x_train, y_train, x_test, y_test)

        save_model(model, params)
    except Exception as e:
        raise Exception(f"Error in main: {e}")

if __name__ == "__main__":
    main()