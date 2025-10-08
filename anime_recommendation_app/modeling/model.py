import tensorflow as tf
from tensorflow import keras
from keras import layers

class HybridRecommenderNet(tf.keras.Model):

    def __init__(self, num_users, num_anime, num_genres, embedding_size, initiallizer, regularization_strength, **kwargs):
        super(HybridRecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_anime = num_anime
        self.num_genres = num_genres
        self.embedding_size = embedding_size
        
        # User and Anime Embeddings (Collaborative Part)
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer=initiallizer, embeddings_regularizer=keras.regularizers.l2(regularization_strength))
        self.user_bias = layers.Embedding(num_users, 1) 
        self.anime_embedding = layers.Embedding(num_anime, embedding_size, embeddings_initializer=initiallizer, embeddings_regularizer=keras.regularizers.l2(regularization_strength))
        self.anime_bias = layers.Embedding(num_anime, 1) 

        # Genre Embedding (Content Part)
        # Using half the size for genre embeddings (25 units)
        self.genre_embedding = layers.Embedding(num_genres, int(embedding_size / 2), embeddings_initializer=initiallizer, embeddings_regularizer=keras.regularizers.l2(regularization_strength)) 
        
        # Dense layers for non-linear interaction
        # The input size to Dense will be EmbSize(U) + EmbSize(A) + EmbSize(G) = 50 + 50 + 25 = 125
        self.dense_layer = layers.Dense(128, activation='relu') 
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # inputs[:, 0]: user_code, inputs[:, 1]: anime_code, inputs[:, 2]: genre_code
        user_vector = self.user_embedding(inputs[:, 0]) 
        anime_vector = self.anime_embedding(inputs[:, 1]) 
        genre_vector = self.genre_embedding(inputs[:, 2]) # Get genre vector

        user_bias = self.user_bias(inputs[:, 0])
        anime_bias = self.anime_bias(inputs[:, 1])

        # Combine all features (Concatenate for Hybrid Model)
        combined_vectors = tf.concat([user_vector, anime_vector, genre_vector], axis=1)
        
        # Pass concatenated vector through dense layers for non-linear interaction
        dense_output = self.dense_layer(combined_vectors)
        
        # Final output layer adds bias terms to the non-linear prediction
        x = dense_output + user_bias + anime_bias
        
        return self.output_layer(x) 