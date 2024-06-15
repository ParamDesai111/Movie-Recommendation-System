import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensuring nltk resources are donwloaded
nltk.download('punkt')

#Load Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df