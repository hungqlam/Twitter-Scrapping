import snscrape.modules.twitter as sntwitter
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import spacy
import datetime
nltk.download('vader_lexicon')
# Import necessary libraries
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nlp = spacy.load("en_core_web_md")
# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

query = "LSDfi"
# Define a function to get sentiment score
def get_sentiment_score(text):
    return sia.polarity_scores(text)['compound']
df = []

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i > 2000:
        break
    df.append([tweet.id, tweet.date, tweet.source, tweet.place, tweet.user.username, tweet.rawContent, tweet.hashtags, tweet.likeCount])

data = pd.DataFrame(df, columns=['id', 'date', 'source', 'place', 'username', 'content', 'hashtags', 'likes'])

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define a function to preprocess text
def preprocess_text(text):
    tokens = [token for token in word_tokenize(text.lower()) if token.isalpha() and not token.startswith(('http', 'www'))]
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Preprocess content
data['tokens'] = data['content'].apply(preprocess_text)

# Topic modeling using LDA
corpus_dict = Dictionary(data['tokens'])
corpus = [corpus_dict.doc2bow(text) for text in data['tokens']]

# Set the number of topics you want to extract
num_topics = 5

lda_model = LdaModel(corpus, num_topics=num_topics, id2word=corpus_dict, random_state=42)
topics = lda_model.print_topics(num_words=5)
print("LDA topics:")
for topic in topics:
    print(topic)

# Entity recognition using SpaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

data['entities'] = data['content'].apply(extract_entities)
# Get sentiment scores
data['sentiment'] = data['content'].apply(get_sentiment_score)
# Save insights to CSV
data.to_csv('insights.csv', index=False)
