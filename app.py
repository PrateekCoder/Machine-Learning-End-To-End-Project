from flask import Flask, render_template, request
import pandas as pd

# Load the similarity dataframe from a pickle file
similarity_df = pd.read_pickle('similarity_df.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Get the user's movie input from the form
    movie = request.form['movie']

    # Find the index of the movie in the similarity dataframe
    try:
        movie_index = similarity_df.index.get_loc(movie)
    except KeyError:
        return render_template('index.html', error='Movie not found. Please try again.')

    # Get the top 10 most similar movies to the movie
    top_10 = similarity_df.iloc[movie_index].sort_values(ascending=False)[1:11]

    # Return the top 10 recommendations to the user
    return render_template('recommendations.html', movie=movie, top_10=top_10)

if __name__ == '__main__':
    app.run(debug=True)
