import pandas as pd
from twitterscraper import query_tweets
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Define your search term and the date_since date as variables
search_words = "#LSDfi"
date_since = "2023-04-01"

# Use the .Cursor method to get tweets
tweets = query_tweets(search_words, begindate = date_since, limit = 500, lang = 'en')

# Create dataframe
df = pd.DataFrame(t.__dict__ for t in tweets)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define a function to get sentiment score
def get_sentiment_score(text):
    return sia.polarity_scores(text)['compound']

# Get sentiment scores
df['sentiment'] = df['text'].apply(get_sentiment_score)

# Save to CSV
df.to_csv('tweets.csv', index=False)
