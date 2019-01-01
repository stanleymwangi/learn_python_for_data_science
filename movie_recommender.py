import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# get movie lens dataset for movies with rating of 4.0 or higher
movie_data = fetch_movielens(min_rating=4.0)

# display properties of movie_data
print('Training data ~ ', repr(movie_data['train']), end='\n\n')
print('Testing data ~ ', repr(movie_data['test']))

# create recommendation model
# using warp (weighted approximate-rank pairwise) loss function
recommendation_model = LightFM(loss='warp') # warp is both content + collaborative based

# train recommendation model
recommendation_model.fit(movie_data['train'], epochs=30, num_threads=2)

def generate_recommendation(model, data, user_ids):

    # get the number of users and number of movies from our data set
    number_users, number_movies = data['train'].shape

    # create movie recommendation for each user
    for user_id in user_ids:

        # get the movies they have liked previously
        previously_liked_movies = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # predict the movies they would like based on our recommendation model by scoring all the movies
        recommended_movie_scores = recommendation_model.predict(user_id, np.arange(number_movies))

        # rank the movies from most liked to least liked
        top_recommended_movies = data['item_labels'][np.argsort(-recommended_movie_scores)]

        # display results
        print('\nUser {}'.format(user_id))
        print('Based on your previously liked movies such as: ')

        for entry in previously_liked_movies[:3]:
            print(str.format(entry))

        print('\nWe recommend you watch: ')
        for entry in top_recommended_movies[:3]:
            print(str.format(entry))

generate_recommendation(recommendation_model, movie_data, [7, 93, 521])