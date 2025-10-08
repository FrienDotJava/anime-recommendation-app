import tensorflow as tf
from tensorflow import keras
from keras import layers  # ok to keep; layers resolves to tf-keras here

try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="RecSys")
class HybridRecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_anime, num_genres, embedding_size,
                 initializer, regularization_strength, **kwargs):
        super().__init__(**kwargs)

        self.num_users = int(num_users)
        self.num_anime = int(num_anime)
        self.num_genres = int(num_genres)
        self.embedding_size = int(embedding_size)
        self.regularization_strength = float(regularization_strength)

        self.initializer_cfg = tf.keras.initializers.serialize(
            tf.keras.initializers.get(initializer)
        )
        init = tf.keras.initializers.deserialize(self.initializer_cfg)
        reg  = tf.keras.regularizers.l2(self.regularization_strength)

        self.user_embedding  = layers.Embedding(self.num_users,  self.embedding_size, embeddings_initializer=init, embeddings_regularizer=reg)
        self.user_bias       = layers.Embedding(self.num_users,  1)
        self.anime_embedding = layers.Embedding(self.num_anime, self.embedding_size, embeddings_initializer=init, embeddings_regularizer=reg)
        self.anime_bias      = layers.Embedding(self.num_anime, 1)
        self.genre_embedding = layers.Embedding(self.num_genres, self.embedding_size // 2, embeddings_initializer=init, embeddings_regularizer=reg)

        self.dense_layer  = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_vector  = self.user_embedding(inputs[:, 0])
        anime_vector = self.anime_embedding(inputs[:, 1])
        genre_vector = self.genre_embedding(inputs[:, 2])
        user_bias  = self.user_bias(inputs[:, 0])
        anime_bias = self.anime_bias(inputs[:, 1])
        combined = tf.concat([user_vector, anime_vector, genre_vector], axis=1)
        dense_out = self.dense_layer(combined)
        x = dense_out + user_bias + anime_bias
        return self.output_layer(x)

    def get_config(self):
        base = super().get_config()
        base.update({
            "num_users": self.num_users,
            "num_anime": self.num_anime,
            "num_genres": self.num_genres,
            "embedding_size": self.embedding_size,
            "initializer": self.initializer_cfg,
            "regularization_strength": self.regularization_strength,
        })
        return base

    @classmethod
    def from_config(cls, config):
        init_cfg = config.pop("initializer")
        initializer = tf.keras.initializers.deserialize(init_cfg)
        return cls(initializer=initializer, **config)
