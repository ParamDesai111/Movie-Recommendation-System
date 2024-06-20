import requests
import json

def fetch_movies():
    # url = "https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page=1&sort_by=popularity.desc"
    url = f"https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page=1&sort_by=popularity.desc"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjMGE2ZDk0NjM1ODcxMDk0MDEyZWYwNDUzZmM2YTVkZSIsInN1YiI6IjY2NmU1MGMxOGVjMDM0Y2IyNTQ3YzQ4MyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.FEPHpvtXFdJf9SHQaMpPlUv7H6oFnDUYPNRawYAm-xs"
    }
    response = requests.get(url, headers=headers)
    return response.json()  # Parse JSON response directly

def display_movies(movies_json):
    results = movies_json.get('results', [])
    print("Movies List:")
    for movie in results:
        title = movie.get('title', 'No Title Available')
        release_date = movie.get('release_date', 'No Release Date')
        overview = movie.get('overview', 'No Overview Available')
        popularity = movie.get('popularity', 0)
        vote_average = movie.get('vote_average', 0)
        print(f"\nTitle: {title}")
        print(f"Release Date: {release_date}")
        print(f"Popularity: {popularity}")
        print(f"Average Vote: {vote_average}")
        print(f"Overview: {overview}")

# Fetch movie data
movies_json = fetch_movies()
# Display the movies
display_movies(movies_json)
