import lyricsgenius as genius
import os
from requests.exceptions import HTTPError, Timeout
import pandas as pd
import numpy as np
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
import nltk.data
import text2emotion as te
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('punkt')

from nrclex import NRCLex


token = '31M-cgnzKCTCgBXvxuvufG5tO-UQoFx0N-Ms4bpqF6QeUacaushTv9WIhkdyNQeV'
#genius = Genius(token, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)
def search_data(query,n,access_token):
    """
    This function uses the library lyricsgenius to extract the fields
    title, artist, album, date and lyrics and stores them into a pandas dataframe
    parameters:
    query = artist or band to search
    n = max numbers of songs
    access_token = your access token of the genius api
    """
    
    api = genius.Genius(access_token,  excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)

    list_lyrics = []
    list_title = []
    list_artist = []
    list_album = []
    list_year = []

    artist = api.search_artist(query,max_songs=n,sort='popularity')
    songs = artist.songs
    for song in songs:
        list_lyrics.append(song.lyrics)
        list_title.append(song.title)
        list_artist.append(song.artist)

    df = pd.DataFrame({'artist':list_artist,'title':list_title, 'lyric':list_lyrics})
    
    return df

def clean_lyrics(df,column):
    """
    This function cleans the words without importance and fix the format of the  dataframe's column lyrics 
    parameters:
    df = dataframe
    column = name of the column to clean
    """
    df = df
    df[column] = df[column].str.lower()
    df[column] = df[column].str.replace("\n"," ")
    df[column] = df[column].str.strip()

    return df

try:
    df = search_data('Taylor Swift', 170, token)
except requests.exceptions.Timeout as e:
    print(e)

df = clean_lyrics(df, 'lyric')
sid = SentimentIntensityAnalyzer()
negative = []
neutral = []
positive = []
compound = []
sentiment_type = []
joy= []
anger = []
surprise = []
sadness = []
fear = []
for i in df.index:
    scores = sid.polarity_scores(df['lyric'].iloc[i])
    #emotions = te.get_emotion(df['lyric'].iloc[i])
    emotions = NRCLex(df['lyric'].iloc[i])
    emotions = emotions.raw_emotion_scores
    negative.append(scores['neg'])
    neutral.append(scores['neu'])
    positive.append(scores['pos'])
    compound.append(scores['compound'])
    try:
        joy.append(emotions['joy'])
    except KeyError:
        joy.append(0)
    try:    
        anger.append(emotions['anger'])
    except KeyError:
        anger.append(0)
    try:
        sadness.append(emotions['sadness'])
    except KeyError:
        sadness.append(0)
    sent = ''
    if scores['compound'] > 0.5:
        sent = 'POSITIVE'
    elif scores['compound'] > -0.5 and scores['compound'] < 0.5 :
        sent = 'NEUTRAL'
    else:
        sent = 'NEGATIVE'
    sentiment_type.append(sent)
    
    
df['negative'] = negative
df['neutral'] = neutral
df['positive'] = positive
df['compound'] = compound
df['sentiment_type'] = sentiment_type
df['joy'] = joy
df['anger'] = anger
df['sadness'] = sadness
df['emotion'] = df[['joy','anger','sadness']].idxmax(axis=1)
df1 = df.drop(columns=['lyric'])
df1.to_json('ts-lyrics.json')

print(df1)