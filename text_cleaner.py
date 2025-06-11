from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))  # Load stopwords once

# Custom Function to apply Text Preprocessing
def transform_text(text):
    text = text.lower()  # Convert text to lowercase

    text = nltk.word_tokenize(text)  # Split text into individual words

    y = []
    # Keep only alphanumeric words (no punctuation or special characters)
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming to reduce words to their base form
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)  # Return cleaned text as a single string

# Custom transformer to apply transform_text to text data
class TextCleaner(BaseEstimator, TransformerMixin):
  def fit(self, X, y = None):
    return self

  def transform(self, X):
    if isinstance(X, pd.Series):
      return X.apply(transform_text)
    elif isinstance(X, np.ndarray):
      return pd.Series(X.ravel()).apply(transform_text)
    else:
      return pd.Series(X).apply(transform_text)
