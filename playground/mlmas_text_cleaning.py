# https://machinelearningmastery.com/clean-text-machine-learning-python/
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import string
import re
from nltk.stem.porter import PorterStemmer

# read data set and convert to big string
df = pd.read_csv('../data/imdb_data.zip')
table = str.maketrans('', '', string.punctuation)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
for i in range(df.shape[0]):
    review = df.iloc[i, 0]  # extract the line
    review = re.sub(r'<[^>]*>', '', review)  # remove (some) html
    words = [w.translate(table) for w in review.split()]  # remove punctuation from words
    words = [w.lower() for w in words]  # to lower case
    words = [ps.stem(w) for w in words]  # stem using Porter
    words = [w for w in words if w not in stop_words and w.isalpha()]  # strip stopwords and numbers

# -----------------------------------------------------------------------------------------------
# NLTK part
# -----------------------------------------------------------------------------------------------
text = df['review'].to_string()
# tokenize words
tokens = word_tokenize(text)
# convert to lower case
tokens = [w.lower() for w in tokens]
# remove punctuation from each word
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
# filter out stop words
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
print(words[:100])
