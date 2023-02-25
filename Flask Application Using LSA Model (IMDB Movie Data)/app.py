from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the IMDb movie data
df = pd.read_csv('imdb_movie_data.csv')

# Combine movie name and tags into a single string
df['content'] = df['Movie'].astype(str) + ' ' + df['runtimeMinutes'].astype(str) + ' ' + df['genres'] + ' ' + df['directors'] + ' ' + df['writers'] + ' ' + df['averageRating'].astype(str) + ' ' + df['numVotes'].astype(str) + df['actors'].astype(str)
df['content'] = df['content'].fillna('')

# Create bag of words
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(df['content'])

# Convert bag of words to TF-IDF
tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(bow)

# Apply LSA or LSI
lsa = TruncatedSVD(n_components=100, algorithm='arpack')
lsa.fit(tfidf)

# Define the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define the recommendation page
@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Get the user input
    user_movie = request.form['movie']

    # Check if the movie is in the database
    try:
        movie_index = df[df['Movie'] == user_movie].index[0]
    except IndexError:
        return render_template('index.html', error='Movie not found. Please try again.')

    # Compute the cosine similarities between the user movie and all other movies
    similarity_scores = cosine_similarity(tfidf[movie_index], tfidf)

    # Get the top 10 most similar movies
    similar_movies = list(enumerate(similarity_scores[0]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:21]

    # Get the movie titles and similarity scores
    top_10 = [(df.loc[i, 'Movie'], score) for i, score in sorted_similar_movies]


    # Render the recommendation page with the recommendations and the user's input
    return render_template('recommendations.html', movie=user_movie, top_10=top_10)



if __name__ == '__main__':
    app.run(debug=True)
