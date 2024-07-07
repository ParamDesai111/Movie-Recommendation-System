import pandas as pd
import numpy as np
import re
import nltk
import requests
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load data from API
def load_movie_from_api(title):
    formatted_title = title.replace(" ", "+")
    url = f"http://www.omdbapi.com/?t={formatted_title}&apikey=d9e6f70e"
    response = requests.get(url)
    if response.status_code == 200:
        movie_data = response.json()
        if movie_data['Response'] == 'True':
            return pd.DataFrame([movie_data])  # Convert single movie data into a DataFrame
        else:
            print("Movie not found")
    else:
        print("Failed to fetch data:", response.status_code)
    return pd.DataFrame()

def process_plot_data(df):
    # Convert plot descriptions to lowercase
    df['clean_plot'] = df['Plot'].str.lower()

    # Remove numbers and punctuation
    df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
    df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('\s+', ' ', x).strip())

    # Tokenize the sentences
    df['clean_plot'] = df['clean_plot'].apply(nltk.word_tokenize)

    # Remove stopwords
    stop_words = nltk.corpus.stopwords.words('english')
    df['clean_plot'] = df['clean_plot'].apply(lambda sentence: [word for word in sentence if word not in stop_words and len(word) >= 3])

    # Join tokens back into a cleaned plot string
    df['clean_plot'] = df['clean_plot'].apply(lambda x: ' '.join(x))

    return df

def clean(sentence):
    temp = []
    for word in sentence:
        temp.append(word.lower().replace(' ',''))
    return temp

def process_other_data(df):
    df['Genre'] = df['Genre'].apply(lambda x: x.split(','))
    df['Actors'] = df['Actors'].apply(lambda x: x.split(',')[:4]) # Getting the top 4 actors
    df['Director'] = df['Director'].apply(lambda x: x.split(',')[:1]) # Getting 1 director
    df['Genre'] = [clean(x) for x in df['Genre']]
    df['Actors'] = [clean(x) for x in df['Actors']]
    df['Director'] = [clean(x) for x in df['Director']]

    columns = ['clean_plot', 'Genre', 'Actors', 'Director']
    l = []

    for i in range(len(df)):
        words = ''
        for col in columns:
            words += ' '.join(df[col][i]) + ' '
        l.append(words)

    df['clean_input'] = l
    df = df[['Title', 'clean_input']]
    df.head() 

    return df

def create_model(df):
    tfidf = CountVectorizer()
    features = tfidf.fit_transform(df['clean_input'])

    cosine_sim = cosine_similarity(features, features)

    return cosine_sim

def get_recommendations(title, cosine_sim, df):
    index = pd.Series(df['Title'])
    movies = [] #movies to recommend
    idx = index[index == title].index[0]

    score = pd.Series(cosine_sim[idx]).sort_values(ascending=False) 
    top10 = list(score.iloc[1:11].index)

    for i in top10:
        movies.append(df['Title'][i])
    return movies

def fetch_sql_data(server, database, username, password):
    driver = 'ODBC Driver 17 for SQL Server'
    connection_string = f"mssql+pyodbc://{username}:{password}@{server}:1433/{database}?driver={driver}"
    engine = create_engine(connection_string)
    query = "SELECT Title, clean_input, Poster FROM cleaned_data"
    df = pd.read_sql(query, engine)
    return df

def combine_data(api_title, server, database, username, password):
    api_df = load_movie_from_api(api_title)
    if not api_df.empty:
        api_df = process_plot_data(api_df)
        api_df = process_other_data(api_df)

    sql_df = fetch_sql_data(server, database, username, password)
    combined_df = pd.concat([api_df, sql_df], ignore_index=True)
    return combined_df
