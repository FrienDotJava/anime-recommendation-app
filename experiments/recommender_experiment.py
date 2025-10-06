import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt

# --- Data Loading ---
# Note: Assuming your 'data/raw/' paths are correct
anime_df = pd.read_csv("data/raw/anime.csv")
rating_df = pd.read_csv("data/raw/rating.csv")

# --- Initial Cleaning ---
rating_df_cleaned = rating_df.drop_duplicates()
rating_df_cleaned.loc[rating_df_cleaned['rating'] == -1, "rating"] = 0
rating_df_cleaned['rating'] = rating_df_cleaned['rating'].values.astype(np.float32)

# ====================================================================
# ✨ FEATURE AUGMENTATION: Pre-process Main Genre on ANIME_DF ✨
# This must happen before any subsets are created.
# ====================================================================

# Calculate 'main_genre' on the master anime DataFrame
anime_df['genre'] = anime_df['genre'].fillna('Unknown')
anime_df['main_genre'] = anime_df['genre'].apply(
    lambda x: x.split(',')[0].strip() if pd.notna(x) and len(x.split(',')) > 0 else 'Unknown'
)

# ====================================================================
# ✨ HYBRID FEATURE EXTRACTION & DATA MERGE (FOR TRAINING) ✨
# ====================================================================

# 1. Merge datasets to get 'type' and 'main_genre' columns
rating_df_merged = pd.merge(rating_df_cleaned, anime_df[['anime_id', 'type', 'main_genre']], on='anime_id')

# 2. Filter by 'TV' series type (Your best filtering practice)
rating_df_cleaned = rating_df_merged[rating_df_merged['type'] == 'TV'].copy()

# Note: We skip the redundant genre extraction since 'main_genre' is already merged.

# ====================================================================
# ✨ DATA FILTERING: REMOVING COLD START USERS AND ITEMS ✨
# (Using optimal thresholds: 100/100)
# ====================================================================

MIN_USER_RATINGS = 100
MIN_ANIME_RATINGS = 100

print(f"Initial filtered TV series ratings: {len(rating_df_cleaned)}.")

# Filter Users (Active Users)
user_counts = rating_df_cleaned['user_id'].value_counts()
active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
rating_df_cleaned = rating_df_cleaned[rating_df_cleaned['user_id'].isin(active_users)]

# Filter Anime (Popular Items)
anime_counts = rating_df_cleaned['anime_id'].value_counts()
popular_anime = anime_counts[anime_counts >= MIN_ANIME_RATINGS].index
rating_df_cleaned = rating_df_cleaned[rating_df_cleaned['anime_id'].isin(popular_anime)]

print(f"Final data size after filtering (100/100): {len(rating_df_cleaned)} ratings.")

# ====================================================================

min_rating = rating_df_cleaned['rating'].min()
max_rating = rating_df_cleaned['rating'].max()

# --- ENCODING ALL FEATURES (User, Anime, Genre) ---

# Encoding User and Anime IDs (Standard)
user_ids = rating_df_cleaned['user_id'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
anime_ids = rating_df_cleaned['anime_id'].unique().tolist()
anime_to_anime_encoded = {x: i for i, x in enumerate(anime_ids)}
anime_encoded_to_anime = {i: x for i, x in enumerate(anime_ids)}

# Encoding Main Genre (New Feature)
genre_names = rating_df_cleaned['main_genre'].unique().tolist()
genre_to_genre_encoded = {x: i for i, x in enumerate(genre_names)}
genre_encoded_to_genre = {i: x for i, x in enumerate(genre_names)}

# Mapping encoded features to the DataFrame
rating_df_cleaned['user'] = rating_df_cleaned['user_id'].map(user_to_user_encoded)
rating_df_cleaned['anime'] = rating_df_cleaned['anime_id'].map(anime_to_anime_encoded)
# New encoded column for the hybrid model input
rating_df_cleaned['genre_code'] = rating_df_cleaned['main_genre'].map(genre_to_genre_encoded)

# Mengacak dataset
rating_df_cleaned = rating_df_cleaned.sample(frac=1, random_state=42)

# --- HYBRID MODEL CLASS ---

class HybridRecommenderNet(tf.keras.Model):

    def __init__(self, num_users, num_anime, num_genres, embedding_size, **kwargs):
        super(HybridRecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_anime = num_anime
        self.num_genres = num_genres
        self.embedding_size = embedding_size
        
        # User and Anime Embeddings (Collaborative Part)
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1) 
        self.anime_embedding = layers.Embedding(num_anime, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.anime_bias = layers.Embedding(num_anime, 1) 

        # Genre Embedding (Content Part)
        # Using half the size for genre embeddings (25 units)
        self.genre_embedding = layers.Embedding(num_genres, int(embedding_size / 2), embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6)) 
        
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

# Pembangunan model
mlflow.set_tracking_uri("https://dagshub.com/FrienDotJava/anime-recommendation-app.mlflow")
mlflow.set_experiment("Experiment on hybrid model")
with mlflow.start_run():
    # Train test split
    # Input 'x' now includes the genre code
    x = rating_df_cleaned[['user', 'anime', 'genre_code']].values

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

    # --- Model Instantiation ---
    regularization_strength = 1e-6
    initializer = 'he_normal'
    
    num_users = len(rating_df_cleaned['user'].unique())
    num_anime = len(rating_df_cleaned['anime'].unique())
    num_genres = len(rating_df_cleaned['main_genre'].unique())

    embedding_size = 50
    learning_rate = 0.0001
    
    # Use the new HybridRecommenderNet and pass num_genres
    model = HybridRecommenderNet(num_users, num_anime, num_genres, embedding_size) 

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
    print(f"Starting training with Hybrid Model. Input features: User, Anime, Genre.")
    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size = batch_size,
        epochs = epoch,
        validation_data = (x_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )


    loss, rmse = model.evaluate(x_val, y_val)
    
    # --- MLflow Logging ---
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
    print(f"\nMLflow logged. Validation RMSE: {rmse:.4f}")
    
# --- Recommendation Logic ---

user_id = rating_df_cleaned.user_id.sample(1).iloc[0]
anime_watched_by_user = rating_df_cleaned[rating_df_cleaned.user_id == user_id]

# 1. Get anime not watched by the user
# 'anime_df' already contains the 'main_genre' column now
anime_not_watched_ids = anime_df[~anime_df['anime_id'].isin(anime_watched_by_user.anime_id.values)]
anime_not_watched_ids = anime_not_watched_ids[anime_not_watched_ids['anime_id'].isin(anime_to_anime_encoded.keys())]

# 2. Filter for only the genres present in our training set and encode features
# This filtering step (which previously caused the KeyError) now works because 'main_genre' exists.
anime_not_watched_ids = anime_not_watched_ids[anime_not_watched_ids['main_genre'].isin(genre_names)].copy()

# Encode the anime and genre features for the prediction array
anime_not_watched_ids['anime_code'] = anime_not_watched_ids['anime_id'].map(anime_to_anime_encoded)
anime_not_watched_ids['genre_code'] = anime_not_watched_ids['main_genre'].map(genre_to_genre_encoded)

# Prepare the final prediction array
anime_not_watched_codes = anime_not_watched_ids[['anime_code', 'genre_code']].values
user_encoder = user_to_user_encoded.get(user_id)

# The prediction array must be N x 3: [User_code, Anime_code, Genre_code]
user_anime_genre_array = np.hstack(
    ([[user_encoder]] * len(anime_not_watched_codes), anime_not_watched_codes)
)

ratings = model.predict(user_anime_genre_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_anime_ids = [
    anime_encoded_to_anime.get(anime_not_watched_codes[x][0]) for x in top_ratings_indices
]

print('\nShowing recommendations for users: {}'.format(user_id))
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
    print(f"{row.name} ({row.anime_id}) : {row.genre}")

print('----' * 8)
print('Top 10 hybrid anime recommendation')
print('----' * 8)

recommended_anime = anime_df[anime_df['anime_id'].isin(recommended_anime_ids)]
for row in recommended_anime.itertuples():
    print(f"{row.name} ({row.anime_id}) : {row.genre}")
