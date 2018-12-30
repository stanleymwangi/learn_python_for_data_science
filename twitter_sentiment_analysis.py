import config

import tweepy
from textblob import TextBlob

# set up twitter authentication
auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_token_secret)

# create handler for twitter api calls
api = tweepy.API(auth)

# get tweet data for specific search term
tweet_data = api.search("life")


for tweet in tweet_data:

    # extract text for analayis
    tweet_blob = TextBlob(tweet.text)

    # display the tweet and it's sentiment
    print(tweet.text, tweet_blob.sentiment)
