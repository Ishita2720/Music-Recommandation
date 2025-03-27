import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load and filter top tracks by popularity
tracks = pd.read_csv('tracks_transformed.csv')
tracks = tracks.sort_values(by=['popularity'], ascending=False).head(10000)

# Handle NaN values in the 'genres' column
tracks['genres'] = tracks['genres'].fillna('')

# Precompute genre vectors and numerical features
song_vectorizer = CountVectorizer()
genre_matrix = song_vectorizer.fit_transform(tracks['genres'])
numerical_features = tracks.select_dtypes(include=np.number).to_numpy()

def get_similarities(song_name, tracks, genre_matrix, numerical_features):
    # Convert song_name to lowercase for case-insensitive matching
    song_idx = tracks[tracks['name'].str.lower() == song_name.lower()].index.to_numpy()
    
    if len(song_idx) == 0:
        print(f"Sorry, the song '{song_name}' was not found.")
        print("Here are some recommended songs from the dataset:\n")
        
        # Recommend fallback songs
        fallback_recommendations = tracks[['name', 'artists']].drop_duplicates().head(10)
        for i, row in fallback_recommendations.iterrows():
            print(f"{row['name']} by {row['artists']}")
        return None
    
    song_idx = song_idx[0]  # Extract the first valid index
    
    # Check if index is within range
    if song_idx >= genre_matrix.shape[0]:
        print(f"Error: song index {song_idx} is out of range for genre_matrix with shape {genre_matrix.shape}.")
        return None

    # Get genre vector and numerical feature for the song
    text_array1 = genre_matrix[song_idx]  # SAFE indexing now
    num_array1 = numerical_features[song_idx].reshape(1, -1)  

    # Compute similarities
    text_sim = cosine_similarity(text_array1, genre_matrix).flatten()
    num_sim = cosine_similarity(num_array1, numerical_features).flatten()

    # Combine similarity scores
    combined_similarity = text_sim + num_sim

    # Copy `tracks` to avoid modifying the original DataFrame
    data = tracks.copy()
    data['similarity_factor'] = combined_similarity
    
    return data

st.title("🎵 Song Recommendation System")
st.write("Enter a song name to get recommendations:")

song_name = st.text_input("Song Name")


if st.button("Recommend"):
    if tracks[tracks['name'] == song_name].shape[0] == 0:
        st.error("NO song found similar,Here are some selected song:")
        suggestions = tracks.sample(5)['name'].values
        for song in suggestions:
            st.write(f"- {song}")
    else:
        updated_data = get_similarities(song_name, tracks, genre_matrix, numerical_features)
        if updated_data is not None:
            recommendations = updated_data.sort_values(by=['similarity_factor', 'popularity'], ascending=[False, False])
            recommended_songs = recommendations[['name', 'artists']].iloc[1:6]
            st.write("Recommended Songs:")
            for index, row in recommended_songs.iterrows():
                st.write(f"🎶 **{row['name']}** by {row['artists']}")
