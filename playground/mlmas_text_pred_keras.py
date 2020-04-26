# https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/

# ######################################################################################################################
# How to Prepare Text Data for Deep Learning with Keras
# ######################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Some examples using one-click-done functions to prepare text
# ----------------------------------------------------------------------------------------------------------------------
# simple tokenization and encoding using keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import hashing_trick

# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
# integer encode the document
result = one_hot(text, round(vocab_size * 1.3))  # use default hashing
result = hashing_trick(text, round(vocab_size * 1.3), hash_function='md5')  # define your hash function
print(result)

# ----------------------------------------------------------------------------------------------------------------------
# Some examples using one-click-done functions to prepare text
# ----------------------------------------------------------------------------------------------------------------------
from keras.preprocessing.text import Tokenizer
# define 5 documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!']
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
# integer encode documents, result is a document term matrix
encoded_docs = t.texts_to_matrix(docs, mode='count')  # note the different modes including, e.g., TFxIDF
print(encoded_docs)