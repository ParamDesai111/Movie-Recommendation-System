import pandas as pd
import numpy as np
import re
import nltk
import requests
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensuring nltk resources are donwloaded
# nltk.download('punkt')

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



#Load Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def combine_data(api_title, csv_file_path):
    api_df = load_movie_from_api(api_title)
    if not api_df.empty:
        api_df = process_plot_data(api_df)
        api_df = process_other_data(api_df)

    csv_df = load_data(csv_file_path)
    csv_df = process_plot_data(csv_df)
    csv_df = process_other_data(csv_df)

    combined_df = pd.concat([api_df, csv_df], ignore_index=True)
    return combined_df


def process_plot_data(df):
    # Ensure required resources are downloaded

    # nltk.download('punkt')
    # nltk.download('stopwords')

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

    return df

# To clean the sentence all lower and remove all the spaces
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

    #combine all of the preproccesing into another dataframe

    columns = ['clean_plot', 'Genre', 'Actors', 'Director']
    l = []

    for i in range(len(df)):
        words = ''
        for col in columns:
            words += ' '.join(df[col][i]) + ' '
        l.append(words)

    df['clean_input'] = l
    df = df[['Title', 'clean_input']]
    df.head() #Processed with the title with the input from genre, plot, actors and director

    return df

# Feature Extraction
def create_model(df):
    # tfidf = TfidfVectorizer()
    tfidf = CountVectorizer()
    features = tfidf.fit_transform(df['clean_input'])

    cosine_sim = cosine_similarity(features, features)

    return cosine_sim

def get_recommendations(title, cosine_sim, df):
    index = pd.Series(df['Title'])
    movies = [] #movies to reccomend
    idx = index[index == title].index[0]
    #print(idx)

    score = pd.Series(cosine_sim[idx]).sort_values(ascending=False) #higher the score the top movie to reccomend
    top10 = list(score.iloc[1:11].index)#(index locater)
    #print(top10)

    for i in top10:
        movies.append(df['Title'][i])
    return movies


# def main():
#     parser = argparse.ArgumentParser(description="Movie Recommendation Engine")
#     parser.add_argument('--data', type=str, help='Path to the movie dataset CSV file', required=True)
#     parser.add_argument('--title', type=str, help='Movie title for recommendations', required=True)
#     args = parser.parse_args()


#     df = load_data(args.data)
#     df = process_plot_data(df)
#     df = process_other_data(df)
#     cosine_sim = create_model(df)

#     movies = get_recommendations(args.title, cosine_sim, df)

#     print(f"Recommendations for '{args.title}': ")

#     for movie in movies:
#         print(movie)

# def main():
#     parser = argparse.ArgumentParser(description="Movie Recommendation Engine")
#     parser.add_argument('--data', type=str, help='Path to the movie dataset CSV file', required=True)
#     parser.add_argument('--title', type=str, help='Movie title for recommendations', required=True)
#     args = parser.parse_args()

#     # Combine data from API and CSV
#     combined_df = combine_data(args.title, args.data)

#     # Process and create the model from the combined data
#     cosine_sim = create_model(combined_df)

#     # Get recommendations based on the combined data
#     movies = get_recommendations(args.title, cosine_sim, combined_df)

#     print(f"Recommendations for '{args.title}': ")
#     for movie in movies:
#         print(movie)

# if __name__ == "__main__":
#     main()