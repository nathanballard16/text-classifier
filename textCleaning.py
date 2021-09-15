import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np

# load the dataset
# TODO: PLACE DATASET HERE **Must be a PICKLE FILE**

path_df = "PATH TO DATASET"

with open(path_df, 'rb') as data:
    df = pickle.load(data)