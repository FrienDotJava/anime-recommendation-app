import tensorflow as tf
from tensorflow import keras
from anime_recommendation_app.helper import load_data, load_params
from .model import HybridRecommenderNet
from .model_training import split_target
import mlflow
import mlflow.tensorflow
import os
import json
import dagshub
from mlflow.models import infer_signature

dagshub.init(repo_owner='FrienDotJava', repo_name='anime-recommendation-app', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/FrienDotJava/anime-recommendation-app.mlflow")
mlflow.set_experiment("Best Model")

def to_dict(mse : float, rmse : float) -> dict:
    try:
        return {
            'mean_squared_error':mse,
            'root_mean_squared_error': rmse
        }
    except Exception as e:  
        raise Exception(f"Error converting metrics to dict: {e}")
    

def save_metrics(metrics : dict, params : dict) -> None:
    try:
        path = params['model_evaluation']['metrics_path']
        with open(path, 'w') as f:
            json.dump(metrics, f)
    except Exception as e:  
        raise Exception(f"Error saving metrics: {e}")
    

def main():
    try: 
        with mlflow.start_run():
            params = load_params()
            model_path = params['model']['model_path']
            test_set_path = params['data']['test_set_path']

            test_set = load_data(test_set_path)
            x_test, y_test = split_target(test_set)

            model = tf.keras.models.load_model(
                model_path, 
                custom_objects={'HybridRecommenderNet': HybridRecommenderNet}
            )
            loss, rmse = model.evaluate(x_test, y_test)

            pred_example = model.predict(x_test)
            signature = infer_signature(x_test, pred_example)

            metrics_dict = to_dict(loss, rmse)
            os.makedirs('reports', exist_ok=True) 
            save_metrics(metrics_dict,params)

            mlflow.log_param("test_size",params['data_preprocessing']['valid_size'])
            mlflow.log_param("embedding_size",params['model']['embedding_size'])
            mlflow.log_param("regularization_strength",params['model']['regularization_strength'])
            mlflow.log_param("initializer",params['model']['initializer'])
            mlflow.log_param("optimizer",params['model']['optimizer'])
            mlflow.log_param("learning_rate",params['model']['learning_rate'])
            mlflow.log_param("epoch",params['model']['epoch'])
            mlflow.log_param("batch_size", params['model']['batch_size'])
            mlflow.log_param("min_user_ratings",params['data']['min_user_ratings'])
            mlflow.log_param("min_anime_ratings",params['data']['min_anime_ratings'])
            mlflow.log_metric("mean_squared_error", loss)
            mlflow.log_metric("root_mean_squared_error", rmse)

            mlflow.tensorflow.log_model(
                model=model,
                artifact_path="hybrid_recommender_model",
                registered_model_name="HybridAnimeRecommender",
                signature=signature
            )
    except Exception as e:
        raise Exception(f"Error in main: {e}")

if __name__ == "__main__":
    main()