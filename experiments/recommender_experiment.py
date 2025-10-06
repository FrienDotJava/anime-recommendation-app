import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import seaborn as sns
import mlflow

anime_df = pd.read_csv("data/raw/anime.csv")

rating_df = pd.read_csv("data/raw/rating.csv")

anime_df_cleaned = anime_df.dropna()
anime_df_cleaned = anime_df_cleaned[anime_df_cleaned['type'] == 'TV']

rating_df_cleaned = rating_df.drop_duplicates()

rating_df_cleaned.loc[rating_df_cleaned['rating']==-1, "rating"] = 0

rating_df_cleaned['rating'] = rating_df_cleaned['rating'].values.astype(np.float32)

MIN_USER_RATINGS = 100

MIN_ANIME_RATINGS = 100

print(f"Original rating dataset size: {len(rating_df_cleaned)} ratings.")

user_counts = rating_df_cleaned['user_id'].value_counts()
# Identify users who meet the minimum rating threshold
active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
# Filter the main DataFrame
rating_df_cleaned = rating_df_cleaned[rating_df_cleaned['user_id'].isin(active_users)]

print(f"After filtering users (min {MIN_USER_RATINGS} ratings): {len(rating_df_cleaned)} ratings.")

# Filter Anime (Popular Items)
# Count how many ratings each anime received (must be recalculated after user filtering)
anime_counts = rating_df_cleaned['anime_id'].value_counts()
# Identify anime that meet the minimum rating threshold
popular_anime = anime_counts[anime_counts >= MIN_ANIME_RATINGS].index
# Filter the main DataFrame
rating_df_cleaned = rating_df_cleaned[rating_df_cleaned['anime_id'].isin(popular_anime)]

print(f"After filtering anime (min {MIN_ANIME_RATINGS} ratings): {len(rating_df_cleaned)} ratings.")

min_rating = rating_df_cleaned['rating'].min()
max_rating = rating_df_cleaned['rating'].max()

# Encoding user_id

# Mengubah user_id menjadi list tanpa nilai yang sama
user_ids = rating_df_cleaned['user_id'].unique().tolist()

# Melakukan encoding user_id
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}

# Melakukan proses encoding angka ke ke user_id
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

# Encoding anime_id

# Mengubah anime_id menjadi list tanpa nilai yang sama
anime_ids = rating_df_cleaned['anime_id'].unique().tolist()

# Melakukan proses encoding anime_id
anime_to_anime_encoded = {x: i for i, x in enumerate(anime_ids)}

# Melakukan proses encoding angka ke anime_id
anime_encoded_to_anime = {i: x for i, x in enumerate(anime_ids)}

# Mapping kolom user_id dan anime_id ke kolom baru

# Mapping user_id ke dataframe user
rating_df_cleaned['user'] = rating_df_cleaned['user_id'].map(user_to_user_encoded)

# Mapping anime_id ke dataframe anime
rating_df_cleaned['anime'] = rating_df_cleaned['anime_id'].map(anime_to_anime_encoded)

# Mengacak dataset
rating_df_cleaned = rating_df_cleaned.sample(frac=1, random_state=42)

# Pembangunan model
mlflow.set_tracking_uri("https://dagshub.com/FrienDotJava/anime-recommendation-app.mlflow")
mlflow.set_experiment("Find best parameter")
with mlflow.start_run():
    # Train test split
    # Membuat variabel x untuk mencocokkan data user dan anime menjadi satu value
    x = rating_df_cleaned[['user', 'anime']].values

    # Membuat variabel y untuk membuat rating dari hasil
    y = rating_df_cleaned['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

    # Membagi menjadi 80% data train dan 20% data validasi
    valid_size = 0.8
    train_indices = int(valid_size * rating_df_cleaned.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:]
    )

    regularization_strength = 1e-6
    initializer = 'he_normal'
    
    class RecommenderNet(tf.keras.Model):
        def __init__(self, num_users, num_anime, embedding_size, **kwargs):
            super(RecommenderNet, self).__init__(**kwargs)
            self.num_users = num_users
            self.num_anime = num_anime
            self.embedding_size = embedding_size
            
            # 1. Standard Embedding Layers
            self.user_embedding = layers.Embedding( 
                num_users,
                embedding_size,
                embeddings_initializer = 'he_normal',
                embeddings_regularizer = keras.regularizers.l2(1e-6)
            )
            self.user_bias = layers.Embedding(num_users, 1) 
            self.anime_embedding = layers.Embedding(
                num_anime,
                embedding_size,
                embeddings_initializer = 'he_normal',
                embeddings_regularizer = keras.regularizers.l2(1e-6)
            )
            self.anime_bias = layers.Embedding(num_anime, 1) 
            
            # 2. Add Dense Layers for Non-Linearity
            self.dense_layer = layers.Dense(16, activation='relu') # 16 units, ReLU activation
    
            # 3. Final Output Layer
            self.output_layer = layers.Dense(1, activation='sigmoid')
    
    
        def call(self, inputs):
            user_vector = self.user_embedding(inputs[:, 0])
            user_bias = self.user_bias(inputs[:, 0])
            anime_vector = self.anime_embedding(inputs[:, 1])
            anime_bias = self.anime_bias(inputs[:, 1])
    
            # Interaction 1: Linear Matrix Factorization (Dot Product)
            dot_product = tf.tensordot(user_vector, anime_vector, 2)
            
            # Interaction 2: Concatenate and pass through Dense layers (Non-linear)
            # Concatenate the user and item vectors
            combined_vectors = tf.concat([user_vector, anime_vector], axis=1)
            
            # Pass the combined vector through the dense layer
            dense_output = self.dense_layer(combined_vectors)
            
            # The final score combines the bias terms with the non-linear output
            x = dense_output + user_bias + anime_bias
            
            return self.output_layer(x)

    num_users = len(rating_df_cleaned['user'].unique())
    num_anime = len(rating_df_cleaned['anime'].unique())

    embedding_size = 50
    learning_rate = 0.0001
    model = RecommenderNet(num_users, num_anime, embedding_size) 

    # model compile
    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    # Model training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_root_mean_squared_error',
        patience=10,
        min_delta=0.001,
        mode='min',
        restore_best_weights=True
    )

    epoch = 100
    batch_size = 2048
    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size = batch_size,
        epochs = epoch,
        validation_data = (x_val, y_val),
        callbacks=[early_stopping]
    )


    loss, rmse = model.evaluate(x_val, y_val)
    mlflow.log_param("valid_size",valid_size)
    mlflow.log_param("embedding_size",embedding_size)
    mlflow.log_param("regularization_strength",regularization_strength)
    mlflow.log_param("initializer",initializer)
    mlflow.log_param("optimizer","adam")
    mlflow.log_param("learning_rate",learning_rate)
    mlflow.log_param("epoch",epoch)
    mlflow.log_param("batch_size",batch_size)
    mlflow.log_param("min_user_rating",MIN_USER_RATINGS)
    mlflow.log_param("min_anime_rating",MIN_ANIME_RATINGS)
    mlflow.log_metric("mse", loss)
    mlflow.log_metric("rmse", rmse)

user_id = rating_df_cleaned.user_id.sample(1).iloc[0]
anime_watched_by_user = rating_df_cleaned[rating_df_cleaned.user_id == user_id]

anime_not_watched = anime_df[~anime_df['anime_id'].isin(anime_watched_by_user.user_id.values)]['anime_id']
anime_not_watched = list(
    set(anime_not_watched)
    .intersection(set(anime_to_anime_encoded.keys()))
)

anime_not_watched = [[anime_to_anime_encoded.get(x)] for x in anime_not_watched]
user_encoder = user_to_user_encoded.get(user_id)
user_anime_array = np.hstack(
    ([[user_encoder]] * len(anime_not_watched), anime_not_watched)
)

ratings = model.predict(user_anime_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_anime_ids = [
    anime_encoded_to_anime.get(anime_not_watched[x][0]) for x in top_ratings_indices
]

print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Anime with high ratings from user')
print('----' * 8)

top_anime_user = (
    anime_watched_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .anime_id.values
)

anime_df_rows = anime_df[anime_df['anime_id'].isin(top_anime_user)]
for row in anime_df_rows.itertuples():
    print(row.name, ':', row.genre)

print('----' * 8)
print('Top 10 anime recommendation')
print('----' * 8)

recommended_anime = anime_df[anime_df['anime_id'].isin(recommended_anime_ids)]
for row in recommended_anime.itertuples():
    print(row.name, ':', row.genre)