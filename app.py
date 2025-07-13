import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
try:
    df = pd.read_csv(r"C:\Users\Amol\Downloads\music recommendation 1\music.csv")
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path!")
    st.stop()

# Preprocess and vectorize text
tfidf = TfidfVectorizer(stop_words='english')
df['text'] = df['text'].fillna('')  # Fill NaN values
tfidf_matrix = tfidf.fit_transform(df['text'])

# Calculate cosine similarity
similarity = cosine_similarity(tfidf_matrix)

# Function to recommend songs
def recommend(song_name):
    try:
        index = df[df['song'].str.lower() == song_name.lower()].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        
        recommended_songs = []
        for i in distances[1:6]:  # Top 5 recommendations
            recommended_songs.append(df.iloc[i[0]]['song'])
        return recommended_songs
    except IndexError:
        return ["Song not found in the dataset. Please check the spelling."]

# Streamlit UI
st.title("🎵 Music_Recommendation_System.ipynb")

song_input = st.text_input("Enter a Song Name:")

if st.button("Recommend"):
    if song_input:
        recommendations = recommend(song_input)
        st.write("### Recommended Songs:")
        for song in recommendations:
            st.write(f"- {song}")
    else:
        st.warning("Please enter a song name to get recommendations.")
