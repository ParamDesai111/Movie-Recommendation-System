# import pandas as pd
# import numpy as np
# import re
# import nltk
# import requests
# import argparse
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Ensuring nltk resources are donwloaded
# # nltk.download('punkt')

# # Load data from API
# def load_movie_from_api(title):
#     """Fetch movie data from the API and return it as a DataFrame."""
#     formatted_title = title.replace(" ", "+")
#     url = f"http://www.omdbapi.com/?t={formatted_title}&apikey=d9e6f70e"  # Replace 'your_api_key' with your actual API key
#     response = requests.get(url)
#     if response.status_code == 200:
#         movie_data = response.json()
#         if movie_data['Response'] == 'True':
#             movie_df = pd.DataFrame([movie_data])
#             movie_df['title'] = movie_df['Title'].str.lower()  # Ensure title is lowercase
#             movie_df['title'] = movie_df['title'].fillna('')  # Fill missing titles
#             return movie_df[['title', 'Year', 'Genre', 'Director', 'Actors', 'Plot']]
#         else:
#             print("Movie not found")
#     else:
#         print("Failed to fetch data:", response.status_code)
#     return pd.DataFrame()

# def check_columns(df):
#     expected_columns = ['title', 'plot', 'genre', 'cast', 'director']
#     for col in expected_columns:
#         if col not in df.columns:
#             df[col] = ""  # Add missing columns as empty strings to prevent key errors
#     return df


# #Load Dataset
# def load_data(file_path):
#     try:
#         df = pd.read_csv(file_path)
#         return df
#     except Exception as e:
#         print(f"Failed to load the data file: {e}")
#         return pd.DataFrame()

# def process_api_data(api_df):
#     if not api_df.empty:
#         api_df.rename(columns={'Plot': 'plot', 'Genre': 'genre', 'Actors': 'cast', 'Director': 'director'}, inplace=True)
#         api_df = process_plot_data(api_df)
#         api_df = process_other_data(api_df)
#     return api_df


# def combine_data(api_title, csv_file_path):
#     api_df = load_movie_from_api(api_title)
    

#     # combined_df = pd.concat([api_df, csv_df], ignore_index=True)
#     # return combined_df
#     """Combine API data with local CSV data."""

#     api_df = load_movie_from_api(api_title)
#     api_df = process_api_data(api_df)

#     csv_df = load_data(csv_file_path)
#     csv_df = process_plot_data(csv_df)
#     csv_df = process_other_data(csv_df)
#     csv_df = pd.read_csv(csv_file_path)
#     csv_df['title'] = csv_df['title'].str.lower()  # Ensure title is lowercase

#     csv_df['title'] = csv_df['title'].fillna('')  # Fill missing titles

#     combined_df = pd.concat([api_df, csv_df], ignore_index=True)
#     return combined_df


# def process_plot_data(df):
#     # Ensure required resources are downloaded

#     # nltk.download('punkt')
#     # nltk.download('stopwords')

#     if 'plot' in df.columns:
#         # Ensure the plot column is a string and handle NaNs by replacing them with an empty string
#         df['plot'] = df['plot'].fillna('')  # Replace NaN with empty string
#         df['clean_plot'] = df['plot'].str.lower()

#         # Remove numbers and punctuation
#         df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
#         df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('\s+', ' ', x).strip())

#         # Tokenize the sentences
#         df['clean_plot'] = df['clean_plot'].apply(nltk.word_tokenize)

#         # Remove stopwords
#         stop_words = nltk.corpus.stopwords.words('english')
#         df['clean_plot'] = df['clean_plot'].apply(lambda sentence: [word for word in sentence if word not in stop_words and len(word) >= 3])
#     return df

# # To clean the sentence all lower and remove all the spaces
# def clean(sentence):
#     temp = []
#     for word in sentence:
#         temp.append(word.lower().replace(' ',''))
#     return temp

# # def process_other_data(df):
# #     df['genre'] = df['genre'].apply(lambda x: x.split(','))
# #     df['cast'] = df['cast'].apply(lambda x: x.split(',')[:4]) # Getting the top 4 actors
# #     df['director'] = df['director'].apply(lambda x: x.split(',')[:1]) # Getting 1 director
# #     df['genre'] = [clean(x) for x in df['Genre']]
# #     df['actors'] = [clean(x) for x in df['Actors']]
# #     df['director'] = [clean(x) for x in df['Director']]

# #     #combine all of the preproccesing into another dataframe

# #     columns = ['clean_plot', 'Genre', 'Actors', 'Director']
# #     l = []

# #     for i in range(len(df)):
# #         words = ''
# #         for col in columns:
# #             words += ' '.join(df[col][i]) + ' '
# #         l.append(words)

# #     df['clean_input'] = l
# #     df = df[['Title', 'clean_input']]
# #     df.head() #Processed with the title with the input from genre, plot, actors and director

# #     return df

# def process_other_data(df):
#     if all(col in df.columns for col in ['genre', 'cast', 'director']):
#         df['genre'] = df['genre'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
#         df['cast'] = df['cast'].apply(lambda x: x.split(',')[:4] if isinstance(x, str) else [])
#         df['director'] = df['director'].apply(lambda x: x.split(',')[:1] if isinstance(x, str) else [])
#         df['genre'] = [clean(x) for x in df['genre']]
#         df['actors'] = [clean(x) for x in df['cast']]
#         df['director'] = [clean(x) for x in df['director']]
#         columns = ['clean_plot', 'genre', 'actors', 'director']
#         df['clean_input'] = df.apply(lambda row: ' '.join([' '.join(row[col]) for col in columns]), axis=1)
#     return df

# # Feature Extraction
# def create_model(df):
#     # tfidf = TfidfVectorizer()
#     tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
#     features = tfidf.fit_transform(df['clean_input'])
#     cosine_sim = cosine_similarity(features, features)
    
#     # Debug: print some parts of the cosine similarity matrix
#     print("Cosine Similarity Matrix Sample:")
#     # print(cosine_sim[:5, :5])  # Print a small part of the matrix
#     print(cosine_sim[0])
    
#     return cosine_sim


# # def get_recommendations(title, cosine_sim, df):
# #     index = pd.Series(df['Title'])
# #     movies = [] #movies to reccomend
# #     idx = index[index == title].index[0]
# #     #print(idx)

# #     score = pd.Series(cosine_sim[idx]).sort_values(ascending=False) #higher the score the top movie to reccomend
# #     top10 = list(score.iloc[1:11].index)#(index locater)
# #     #print(top10)

# #     for i in top10:
# #         movies.append(df['Title'][i])
# #     return movies
# def get_recommendations(title, cosine_sim, df):
#     try:
#         titles = df['title'].tolist()  # Make sure to use 'title' instead of 'Title'
#         idx = titles.index(title)
#         score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
#         top_indices = list(score.iloc[1:11].index)  # Adjust based on needed output
#         recommended_movies = df.iloc[top_indices]['title'].tolist()  # Using 'title' here
#         return recommended_movies
#     except Exception as e:
#         print(f"Error in generating recommendations: {e}")
#         return []


# # def main():
# #     parser = argparse.ArgumentParser(description="Movie Recommendation Engine")
# #     parser.add_argument('--data', type=str, help='Path to the movie dataset CSV file', required=True)
# #     parser.add_argument('--title', type=str, help='Movie title for recommendations', required=True)
# #     args = parser.parse_args()


# #     df = load_data(args.data)
# #     df = process_plot_data(df)
# #     df = process_other_data(df)
# #     cosine_sim = create_model(df)

# #     movies = get_recommendations(args.title, cosine_sim, df)

# #     print(f"Recommendations for '{args.title}': ")

# #     for movie in movies:
# #         print(movie)

# # def main():
# #     parser = argparse.ArgumentParser(description="Movie Recommendation Engine")
# #     parser.add_argument('--data', type=str, help='Path to the movie dataset CSV file', required=True)
# #     parser.add_argument('--title', type=str, help='Movie title for recommendations', required=True)
# #     args = parser.parse_args()

# #     # Combine data from API and CSV
# #     combined_df = combine_data(args.title, args.data)

# #     # Process and create the model from the combined data
# #     cosine_sim = create_model(combined_df)

# #     # Get recommendations based on the combined data
# #     movies = get_recommendations(args.title, cosine_sim, combined_df)

# #     print(f"Recommendations for '{args.title}': ")
# #     for movie in movies:
# #         print(movie)

# # if __name__ == "__main__":
# #     main()

# # def main():
# #     parser = argparse.ArgumentParser(description="Movie Recommendation Engine")
# #     parser.add_argument('--data', type=str, help='Path to the movie dataset CSV file', required=True)
# #     parser.add_argument('--title', type=str, help='Movie title for recommendations', required=True)
# #     args = parser.parse_args()

# #     combined_df = combine_data(args.title, args.data)
# #     combined_df = check_columns(combined_df)  # Check and correct columns after loading and combining data

# #     if not combined_df.empty:
# #         combined_df = process_plot_data(combined_df)
# #         combined_df = process_other_data(combined_df)
# #         cosine_sim = create_model(combined_df)
# #         movies = get_recommendations(args.title, cosine_sim, combined_df)

# #         print(f"Recommendations for '{args.title}':")
# #         for movie in movies:
# #             print(movie)
# #     else:
# #         print("No data available for processing.")

# # if __name__ == "__main__":
# #     main()


# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description="Movie Recommendation Engine")
#     parser.add_argument('--data', type=str, help='Path to the movie dataset CSV file', required=True)
#     parser.add_argument('--title', type=str, help='Movie title for recommendations', required=True)
#     args = parser.parse_args()

#     combined_df = combine_data(args.title, args.data)
#     if not combined_df.empty:
#         print("Data combined successfully. Proceed with processing.")
#         combined_df = process_plot_data(combined_df)
#         combined_df = process_other_data(combined_df)
#         cosine_sim = create_model(combined_df)
#         movies = get_recommendations(args.title, cosine_sim, combined_df)

#         print(f"Recommendations for '{args.title}':")
#         for movie in movies:
#             print(movie)

#     else:
#         print("No data available for processing.")

# if __name__ == "__main__":
#     main()

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
            print_movie_details(movie_data)
            return pd.DataFrame([movie_data])  # Convert single movie data into a DataFrame
        else:
            print("Movie not found")
    else:
        print("Failed to fetch data:", response.status_code)
    return pd.DataFrame()

def print_movie_details(movie):
    print("Movie Details:")
    print(f"Title: {movie.get('Title', 'N/A')}")
    print(f"Genre: {movie.get('Genre', 'N/A')}")
    print(f"Director: {movie.get('Director', 'N/A')}")
    print(f"Actors: {movie.get('Actors', 'N/A')}")
    print(f"Plot: {movie.get('Plot', 'N/A')}")
    print(f"Poster: {movie.get('Poster', 'N/A')}")


def check_columns(df):
    expected_columns = ['title', 'plot', 'genre', 'cast', 'director']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""  # Add missing columns as empty strings to prevent key errors
    return df


#Load Dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Failed to load the data file: {e}")
        return pd.DataFrame()

def process_api_data(api_df):
    if not api_df.empty:
        api_df.rename(columns={'Plot': 'plot', 'Genre': 'genre', 'Actors': 'cast', 'Director': 'director'}, inplace=True)
        api_df = process_plot_data(api_df)
        api_df = process_other_data(api_df)
    return api_df


def combine_data(api_title, csv_file_path):
    api_df = load_movie_from_api(api_title)
    api_df = process_api_data(api_df)

    csv_df = load_data(csv_file_path)
    csv_df = process_plot_data(csv_df)
    csv_df = process_other_data(csv_df)

    combined_df = pd.concat([api_df, csv_df], ignore_index=True)
    return combined_df


def process_plot_data(df):
    # Ensure required resources are downloaded

    # nltk.download('punkt')
    # nltk.download('stopwords')

    if 'plot' in df.columns:
        # Ensure the plot column is a string and handle NaNs by replacing them with an empty string
        df['plot'] = df['plot'].fillna('')  # Replace NaN with empty string
        df['clean_plot'] = df['plot'].str.lower()

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

# def process_other_data(df):
#     df['genre'] = df['genre'].apply(lambda x: x.split(','))
#     df['cast'] = df['cast'].apply(lambda x: x.split(',')[:4]) # Getting the top 4 actors
#     df['director'] = df['director'].apply(lambda x: x.split(',')[:1]) # Getting 1 director
#     df['genre'] = [clean(x) for x in df['Genre']]
#     df['actors'] = [clean(x) for x in df['Actors']]
#     df['director'] = [clean(x) for x in df['Director']]

#     #combine all of the preproccesing into another dataframe

#     columns = ['clean_plot', 'Genre', 'Actors', 'Director']
#     l = []

#     for i in range(len(df)):
#         words = ''
#         for col in columns:
#             words += ' '.join(df[col][i]) + ' '
#         l.append(words)

#     df['clean_input'] = l
#     df = df[['Title', 'clean_input']]
#     df.head() #Processed with the title with the input from genre, plot, actors and director

#     return df

def process_other_data(df):
    if all(col in df.columns for col in ['genre', 'cast', 'director']):
        df['genre'] = df['genre'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        df['cast'] = df['cast'].apply(lambda x: x.split(',')[:4] if isinstance(x, str) else [])
        df['director'] = df['director'].apply(lambda x: x.split(',')[:1] if isinstance(x, str) else [])
        df['genre'] = [clean(x) for x in df['genre']]
        df['actors'] = [clean(x) for x in df['cast']]
        df['director'] = [clean(x) for x in df['director']]
        columns = ['clean_plot', 'genre', 'actors', 'director']
        df['clean_input'] = df.apply(lambda row: ' '.join([' '.join(row[col]) for col in columns]), axis=1)
    return df

# Feature Extraction
def create_model(df):
    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(df['clean_input'])
    cosine_sim = cosine_similarity(features, features)
    
    # Debug: print some parts of the cosine similarity matrix
    print("Cosine Similarity Matrix Sample:")
    # print(cosine_sim[:5, :5])  # Print a small part of the matrix
    print(cosine_sim)
    
    return cosine_sim


# def get_recommendations(title, cosine_sim, df):
#     index = pd.Series(df['Title'])
#     movies = [] #movies to reccomend
#     idx = index[index == title].index[0]
#     #print(idx)

#     score = pd.Series(cosine_sim[idx]).sort_values(ascending=False) #higher the score the top movie to reccomend
#     top10 = list(score.iloc[1:11].index)#(index locater)
#     #print(top10)

#     for i in top10:
#         movies.append(df['Title'][i])
#     return movies
def get_recommendations(title, cosine_sim, df):
    try:
        titles = df['title'].tolist()  # Make sure to use 'title' instead of 'Title'
        idx = titles.index(title)
        score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
        top_indices = list(score.iloc[1:11].index)  # Adjust based on needed output
        recommended_movies = df.iloc[top_indices]['title'].tolist()  # Using 'title' here
        print(f"Recommendations for '{title}':")
        for movie in recommended_movies:
            print(movie)
        return recommended_movies
    except Exception as e:
        print(f"Error in generating recommendations: {e}")
        return []


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

def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation Engine")
    parser.add_argument('--data', type=str, help='Path to the movie dataset CSV file', required=True)
    parser.add_argument('--title', type=str, help='Movie title for recommendations', required=True)
    args = parser.parse_args()

    combined_df = combine_data(args.title, args.data)
    combined_df = check_columns(combined_df)  # Check and correct columns after loading and combining data

    if not combined_df.empty:
        combined_df = process_plot_data(combined_df)
        combined_df = process_other_data(combined_df)
        cosine_sim = create_model(combined_df)
        movies = get_recommendations(args.title, cosine_sim, combined_df)

        print(f"Recommendations for '{args.title}':")
        for movie in movies:
            print(movie)
    else:
        print("No data available for processing.")

if __name__ == "__main__":
    main()
