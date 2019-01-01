import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# get movie lens dataset for movies with rating of 4.0 or higher
movie_data = fetch_movielens(min_rating=4.0)

# display properties of movie_data
print("Training data ~ ", repr(movie_data["train"]), end="\n\n")
print("Testing data ~ ", repr(movie_data["test"]))