from flask import Flask, request, jsonify, request, render_template
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


app = Flask(__name__)



def load_data():
    movies_df = pd.read_csv('movies.csv')
    ratings_df = pd.read_csv('ratings.csv')
    duplicates = movies_df['title'].value_counts()
    duplicates = duplicates[duplicates > 1].sort_values(ascending=False)

    def remove_year_from_title(title):
        return re.sub(r'\s\(\d{4}\)$', '', title)
    
    duplicate_movies_list = duplicates.index.tolist()
    duplicate_movies_list

    movies_df = movies_df[movies_df['movieId'] != 168358]
    movies_df = movies_df[movies_df['movieId'] != 144606]
    movies_df = movies_df[movies_df['movieId'] != 26958]
    movies_df = movies_df[movies_df['movieId'] != 64997]
    movies_df = movies_df[movies_df['movieId'] != 147002]

    movie_ratings_df = pd.merge(movies_df, ratings_df, on='movieId')

    movie_ratings_df = movie_ratings_df[['userId','movieId', 'title', 'genres', 'rating']]

    movie_ratings_df.sort_values(['userId','movieId'], inplace=True)

    movie_ratings_df.reset_index(drop=True, inplace=True)

    movie_ratings_df['title'] = movie_ratings_df['title'].str.strip()
    movie_ratings_df['genres'] = movie_ratings_df['genres'].str.strip()
    movie_ratings_df['year'] = movie_ratings_df['title'].str[-5:-1]

    strings_to_drop = ['irro', 'atso', ' Bab', 'ron ', 'r On', 'lon ', 'imal', 'osmo', 'he O', ' Roa', 'ligh', 'erso']

    for string in strings_to_drop:
        movie_ratings_df = movie_ratings_df[~movie_ratings_df['year'].str.contains(string)]

    movie_ratings_df[movie_ratings_df['genres']=='(no genres listed)'].drop_duplicates('movieId')['movieId'].count()

    movie_ratings_df = movie_ratings_df[movie_ratings_df['genres'] != "(no genres listed)"]

    movie_ratings_df['year'] = movie_ratings_df['year'].astype(int)

    year_column_type = movie_ratings_df['year'].dtype

    genre_df = movie_ratings_df[['genres']]

    genre_df = genre_df['genres'].str.split('|', expand=True)

    def genre_name(dataframe):
        df = dataframe.copy()
        col = df.columns
        u = set()
        for i in col:
            s = set(df[i].value_counts().index)
            u = u.union(s)
        return(u)
    
    gens = genre_name(genre_df)

    for genre in gens:
        movie_ratings_df[genre] = movie_ratings_df['genres'].apply(lambda x: 1 if genre in x else 0)

    return movie_ratings_df

def recommend_movies_for_user(movie_ratings_df, user_id):
    # Step 1: Filter for the specified user and calculate weights
    user_df = movie_ratings_df[movie_ratings_df['userId'] == user_id]
    user_ratings_df = user_df['rating']
    user_movie_df = user_df.iloc[:, 6:]
    user_movie_weights_df = user_movie_df.multiply(user_ratings_df, axis=0)
    wgn = pd.concat((user_df.iloc[:, :6], user_movie_weights_df), axis=1)
    wg = wgn.iloc[:, 6:].sum(axis=0) / wgn.iloc[:, 6:].sum(axis=0).sum()
    
    all_movies_id = set(movie_ratings_df[movie_ratings_df['userId'] == user_id]['movieId'].values)
    user_movies_id = set(movie_ratings_df['movieId'].unique())
    user_unwatched_movie_id = user_movies_id - all_movies_id
    foreign_users = movie_ratings_df[movie_ratings_df['userId'] != user_id]
    unwatched_movies = foreign_users[foreign_users['movieId'].isin(user_unwatched_movie_id)].drop(['userId', 'rating'], axis=1)
    
    unwatched_movies_with_weights = pd.concat((unwatched_movies.iloc[:, :4], unwatched_movies.iloc[:, 4:].multiply(wg, axis=1)), axis=1)
    unwatched_movies_with_weights["final_score"] = unwatched_movies_with_weights.iloc[:, 4:].sum(axis=1)
    unwatched_movies_with_weights.sort_values('final_score', ascending=False, inplace=True)
    
    filtered_movies = unwatched_movies_with_weights[unwatched_movies_with_weights["genres"].str.contains("|", regex=False)]
    sorted_movies = filtered_movies.sort_values(by='final_score', ascending=False)
    top_10_recommended_movies = sorted_movies.head(10)
    unique_top_10_movies_titles = set(top_10_recommended_movies['title'])
    
    return unique_top_10_movies_titles


@app.route('/', methods=['GET', 'POST'])
def home():
    current_year = datetime.now().year  
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        movie_ratings_df = load_data()  
        recommended_movies = recommend_movies_for_user(movie_ratings_df, user_id)  
        return render_template('index.html', recommended_movies=recommended_movies, user_id=user_id, current_year=current_year)
    else:
        return render_template('index.html', recommended_movies=None, user_id=None, current_year=current_year)

@app.route('/user_id/<int:user_id>', methods=['GET'])
def user_recommendations(user_id):
    current_year = datetime.now().year  
    movie_ratings_df = load_data()  
    recommended_movies = recommend_movies_for_user(movie_ratings_df, user_id)  
    return render_template('index.html', recommended_movies=recommended_movies, user_id=user_id, current_year=current_year)


if __name__ == '__main__':
    app.run(debug=True)