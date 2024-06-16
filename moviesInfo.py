import requests
import json

def fetch_movies():
    url = "http://www.omdbapi.com/?t=The+Godfather&apikey=d9e6f70e"
    response = requests.get(url)
    if response.status_code == 200:
        movie_data = response.json()  # Parse the JSON response
        print_movie_details(movie_data)
    else:
        print("Failed to fetch data:", response.status_code)

def print_movie_details(movie):
    print("Movie Details:")
    print(f"Title: {movie.get('Title', 'N/A')}")
    print(f"Year: {movie.get('Year', 'N/A')}")
    print(f"Rated: {movie.get('Rated', 'N/A')}")
    print(f"Released: {movie.get('Released', 'N/A')}")
    print(f"Runtime: {movie.get('Runtime', 'N/A')}")
    print(f"Genre: {movie.get('Genre', 'N/A')}")
    print(f"Director: {movie.get('Director', 'N/A')}")
    print(f"Writer: {movie.get('Writer', 'N/A')}")
    print(f"Actors: {movie.get('Actors', 'N/A')}")
    print(f"Plot: {movie.get('Plot', 'N/A')}")
    print(f"Language: {movie.get('Language', 'N/A')}")
    print(f"Country: {movie.get('Country', 'N/A')}")
    print(f"Awards: {movie.get('Awards', 'N/A')}")
    print(f"Poster: {movie.get('Poster', 'N/A')}")
    print(f"Ratings: {format_ratings(movie.get('Ratings', []))}")
    print(f"Metascore: {movie.get('Metascore', 'N/A')}")
    print(f"IMDB Rating: {movie.get('imdbRating', 'N/A')}")
    print(f"IMDB Votes: {movie.get('imdbVotes', 'N/A')}")
    print(f"IMDB ID: {movie.get('imdbID', 'N/A')}")
    print(f"Box Office: {movie.get('BoxOffice', 'N/A')}")
    print(f"Production: {movie.get('Production', 'N/A')}")

def format_ratings(ratings):
    return ', '.join([f"{rating['Source']}: {rating['Value']}" for rating in ratings])

# Fetch movie data
fetch_movies()
