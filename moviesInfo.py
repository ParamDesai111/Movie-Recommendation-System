import requests
import json
import pandas as pd

def fetch_movies():
    url = "http://www.omdbapi.com/?t=Ant-Man&apikey=d9e6f70e"
    response = requests.get(url)
    # if response.status_code == 200:
    #     movie_data = response.json()  # Parse the JSON response
    #     print_movie_details(movie_data)
    # else:
    #     print("Failed to fetch data:", response.status_code)
    if response.status_code == 200:
        movie_data = response.json()
        if movie_data['Response'] == 'True':
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

# Fetch movie data
df = fetch_movies()

print(df['Title'])
print(df['Genre'])
print(df['Actors'])
print(df['Plot'])
