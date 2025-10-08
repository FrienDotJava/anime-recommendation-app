import tensorflow as tf
from tensorflow import keras
from helper import load_data, load_params
from model import HybridRecommenderNet
from model_training import split_target
import mlflow
import mlflow.tensorflow
import os

mlflow.set_tracking_uri("https://dagshub.com/FrienDotJava/anime-recommendation-app.mlflow")
mlflow.set_experiment("Best Model")
os.environ["MLFLOW_TRACKING_USERNAME"] = "FrienDotJava"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "245a82155397ddc77977d0d01542f7eb9209d28d"

def main():
    try: 
        with mlflow.start_run():
            params = load_params()
            model_path = params['model']['model_path']
            valid_set_path = params['data']['valid_set_path']

            valid_set = load_data(valid_set_path)
            x_valid, y_valid = split_target(valid_set)

            model = tf.keras.models.load_model(
                model_path, 
                custom_objects={'HybridRecommenderNet': HybridRecommenderNet}
            )
            loss, rmse = model.evaluate(x_valid, y_valid)

            mlflow.log_param("valid_size",params['data_preprocessing']['valid_size'])
            mlflow.log_param("embedding_size",params['model']['embedding_size'])
            mlflow.log_param("regularization_strength",params['model']['regularization_strength'])
            mlflow.log_param("initializer",params['model']['initializer'])
            mlflow.log_param("optimizer",params['model']['optimizer'])
            mlflow.log_param("learning_rate",params['model']['learning_rate'])
            mlflow.log_param("epoch",params['model']['epoch'])
            mlflow.log_param("batch_size", params['model']['batch_size'])
            mlflow.log_param("min_user_rating",params['data']['min_user_ratings'])
            mlflow.log_param("min_anime_rating",params['data']['min_anime_rating'])
            mlflow.log_metric("mse", loss)
            mlflow.log_metric("rmse", rmse)

            mlflow.tensorflow.log_model(
                tf_model=model,
                artifact_path="hybrid_recommender_model",
                registered_model_name="HybridAnimeRecommender"
            )
    except Exception as e:
        raise Exception(f"Error in main: {e}")

if __name__ == "__main__":
    main()