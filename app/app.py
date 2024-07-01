from flask import Flask, request, render_template
from recommendations_system.model import combine_data, create_model, get_recommendations

app = Flask(__name__)

# Load and process data once on startup
csv_file_path = 'IMDB_Top250Engmovies2_OMDB_Detailed.csv'  # The correct path to your CSV file in the root directory
combined_df = combine_data("", csv_file_path)
cosine_sim = create_model(combined_df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['title']
    if movie_title:
        combined_df = combine_data(movie_title, csv_file_path)
        cosine_sim = create_model(combined_df)
        recommendations = get_recommendations(movie_title, cosine_sim, combined_df)
    else:
        recommendations = ["Please provide a valid movie title."]
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, render_template
# from recommendations_system.largerModel import combine_data, create_model, get_recommendations, check_columns, process_plot_data, process_other_data

# app = Flask(__name__)

# # Path to the CSV file
# csv_file_path = 'movies_initial.csv'

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     # movie_title = request.form['title'].strip().lower()  # Ensure the title is lowercase and stripped of extra spaces
#     movie_title = request.form['title']
#     if movie_title:
#         combined_df = combine_data(movie_title, csv_file_path)
#         combined_df = check_columns(combined_df)
#         combined_df = process_plot_data(combined_df)
#         combined_df = process_other_data(combined_df)
#         cosine_sim = create_model(combined_df)
#         recommendations = get_recommendations(movie_title, cosine_sim, combined_df)
#     else:
#         recommendations = ["Please provide a valid movie title."]
#     return render_template('recommendations.html', recommendations=recommendations)

# if __name__ == '__main__':
#     app.run(debug=True)

