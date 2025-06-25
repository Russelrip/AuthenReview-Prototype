# preprocessing.py

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
import re


class TextCleaner(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = []
        for doc in X:
            text = re.sub(r'\W', ' ', str(doc))
            text = re.sub(r'\s+', ' ', text).lower()
            tokens = [
                w for w in text.split()
                if w not in self.stop_words
            ]
            cleaned.append(' '.join(tokens))
        return cleaned

