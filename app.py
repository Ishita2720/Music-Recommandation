import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from difflib import get_close_matches  # For fuzzy matching

# Load and filter top tracks by popularity
tracks = pd.read_csv('tracks_transformed.csv')
tracks = tracks.sort_values(by=['popularity'], ascending=False).head(10000)

# Handle NaN values and normalize song names (lowercase and strip spaces)
tracks['genres'] = tracks['genres'].fillna('')
tracks['name'] = tracks['name'].str.lower().str.strip()

# Precompute genre vectors and numerical features
song_vectorizer = CountVectorizer()
genre_matrix = song_vectorizer.fit_transform(tracks['genres'])
numerical_features = tracks.select_dtypes(include=np.number).to_numpy()

def get_similarities(song_name, data, genre_matrix, numerical_features):
    # Normalize the input song name
    song_name = song_name.lower().strip()

    # Find exact or closest match if song is not found
    if song_name not in data['name'].values:
        closest_matches = get_close_matches(song_name, data['name'].values, n=1, cutoff=0.7)
        if closest_matches:
            song_name = closest_matches[0]  # Use the closest match
            st.warning(f"Did you mean **{song_name}**? Using this for recommendations.")
        else:
            return None

    # Get the index of the song
    song_idx = data[data['name'] == song_name].index[0]
    text_array1 = genre_matrix[song_idx]
    num_array1 = numerical_features[song_idx].reshape(1, -1)

    text_sim = cosine_similarity(text_array1, genre_matrix).flatten()
    num_sim = cosine_similarity(num_array1, numerical_features).flatten()

    combined_similarity = text_sim + num_sim
    data['similarity_factor'] = combined_similarity
    return data

# Streamlit app layout
st.title("🎵 Song Recommendation System")
st.write("Enter a song name to get recommendations:")

song_name = st.text_input("Song Name")

if st.button("Recommend"):
    if not song_name.strip():  # Handle empty input
        st.error("Please enter a valid song name.")
    else:
        updated_data = get_similarities(song_name, tracks, genre_matrix, numerical_features)
        
        if updated_data is None:
            st.error("Song not found! Here are some random suggestions:")
            suggestions = tracks.sample(5)['name'].values
            for song in suggestions:
                st.write(f"- {song}")
        else:
            recommendations = updated_data.sort_values(by=['similarity_factor', 'popularity'], ascending=[False, False])
            recommended_songs = recommendations[['name', 'artists']].iloc[1:6]
            st.write("Recommended Songs:")
            for index, row in recommended_songs.iterrows():
                st.write(f"🎶 **{row['name'].title()}** by {row['artists']}")
