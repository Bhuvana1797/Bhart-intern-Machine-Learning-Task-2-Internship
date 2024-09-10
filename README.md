Introduction:
This project implements a content-based movie recommendation system using Python. The system leverages features from the movie dataset, such as genre and rating, to provide personalized recommendations. By analyzing movie genres with TF-IDF Vectorization and incorporating ratings, the system identifies and suggests movies similar to a given title. This approach enhances the movie-watching experience by helping users discover films they are likely to enjoy based on their preferences.
import pandas as pd
# Display the first few rows to understand the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing values if necessary
df['genre'] = df['genre'].fillna('')

### 3. Data Preprocessing


from sklearn.feature_extraction.text import TfidfVectorizer

# Process genres: replace '|' with spaces if necessary
df['genre'] = df['genre'].apply(lambda x: ' '.join(x.split('|')))

# Initialize TF-IDF Vectorizer for genres
genre_vectorizer = TfidfVectorizer()
genre_matrix = genre_vectorizer.fit_transform(df['genre'])


### 4. Compute Similarity

#Calculate similarity between movies based on genre:


from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity between movies based on genres
genre_similarity = cosine_similarity(genre_matrix, genre_matrix)

### 5. Build the Recommendation Function

#Create a function to recommend movies based on a given movie name:


def get_recommendations(movie_name, df, similarity_matrix, top_n=10):
    """
    Get movie recommendations based on a given movie name.
    
    Parameters:
        movie_name (str): The name of the movie to base recommendations on.
        df (DataFrame): The DataFrame containing movie data.
        similarity_matrix (array): The matrix of similarities between movies.
        top_n (int): The number of top recommendations to return.
        
    Returns:
        list: A list of recommended movie names.
    """
    # Ensure the movie name exists in the dataset
    if movie_name not in df['name'].values:
        return "Movie name not found in the dataset."
    
    # Get index of the movie
    idx = df[df['name'] == movie_name].index[0]
    
    # Get pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get scores of the top_n most similar movies
    sim_scores = sim_scores[1:top_n+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    return df['name'].iloc[movie_indices].tolist()

# Example usage
recommendations = get_recommendations('The Shining', df, genre_similarity)
print(recommendations)

