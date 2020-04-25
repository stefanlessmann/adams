# https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281
# https://github.com/nathanrooy/word2vec-from-scratch-with-python/blob/master/word2vec.py
# https://nathanrooy.github.io/posts/2018-03-22/word2vec-from-scratch-with-python-and-numpy/
from collections import defaultdict

import numpy as np

text = "natural language processing and machine learning is fun and exciting"

# Note the .lower() as upper and lowercase does not matter in our implementation
# [['natural', 'language', 'processing', 'and', 'machine', 'learning', 'is', 'fun', 'and', 'exciting']]
corpus = [[word.lower() for word in text.split()]]

settings = {
    'window_size': 2,  # context window +- center word
    'n': 10,  # dimensions of word embeddings, also refer to size of hidden layer
    'epochs': 50,  # number of training epochs
    'learning_rate': 0.01  # learning rate
}


class Word2Vec:
    def __init__(self):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']

    def generate_training_data(self, settings, corpus):
        # Find unique word counts using dictonary
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1
        ## How many unique words in vocab? 9
        self.v_count = len(word_counts.keys())
        # Generate Lookup Dictionaries (vocab)
        self.words_list = list(word_counts.keys())
        # Generate word:index
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        # Generate index:word
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        # Cycle through each sentence in corpus
        for sentence in corpus:
            sent_len = len(sentence)
            # Cycle through each word in sentence
            for i, word in enumerate(sentence):
                # Convert target word to one-hot
                w_target = self.word2onehot(sentence[i])
                # Cycle through context window
                w_context = []
                # Note: window_size 2 will have range of 5 values
                for j in range(i - self.window, i + self.window + 1):
                    # Criteria for context word
                    # 1. Target word cannot be context word (j != i)
                    # 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
                    # 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range
                    if j != i and j <= sent_len - 1 and j >= 0:
                        # Append the one-hot representation of word to w_context
                        w_context.append(self.word2onehot(sentence[j]))
                        # print(sentence[i], sentence[j])
                        # training_data contains a one-hot representation of the target word and context words
                training_data.append([w_target, w_context])
        return np.array(training_data)

    def word2onehot(self, word):
        # word_vec - initialise a blank vector
        word_vec = [0 for i in range(0, self.v_count)]  # Alternative - np.zeros(self.v_count)
        # Get ID of word from word_index
        word_index = self.word_index[word]
        # Change value from 0 to 1 according to ID of the word
        word_vec[word_index] = 1
        return word_vec

    def train(self, training_data):
        # Initialising weight matrices
        # Both s1 and s2 should be randomly initialised but for this demo, we pre-determine the arrays (getW1 and getW2)
        # getW1 - shape (9x10) and getW2 - shape (10x9)
        # self.w1 = np.array(getW1)
        # self.w2 = np.array(getW2)
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
        # Cycle through each epoch
        for i in range(self.epochs):
            # Intialise loss to 0
            self.loss = 0

            # Cycle through each training sample
            # w_t = vector for target word, w_c = vectors for context words
            for w_t, w_c in training_data:
                # Forward pass - Pass in vector for target word (w_t) to get:
                # 1. predicted y using softmax (y_pred) 2. matrix of hidden layer (h) 3. output layer before softmax (u)
                y_pred, h, u = self.forward_pass(w_t)

                # Calculate error
                # 1. For a target word, calculate difference between y_pred and each of the context words
                # 2. Sum up the differences using np.sum to give us the error for this particular target word
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # Backpropagation
                # We use SGD to backpropagate errors - calculate loss on the output layer
                self.backprop(EI, h, w_t)

                # Calculate loss
                # There are 2 parts to the loss function
                # Part 1: -ve sum of all the output +
                # Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
                # Note: word.index(1) returns the index in the context word vector with value 1
                # Note: u[word.index(1)] returns the value of the output layer before softmax
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
            print('Epoch:', i, "Loss:", self.loss)


    def forward_pass(self, x):
        # x is one-hot vector for target word, shape - 9x1
        # Run through first matrix (w1) to get hidden layer - 10x9 dot 9x1 gives us 10x1
        h = np.dot(self.w1.T, x)
        # Dot product hidden layer with second matrix (w2) - 9x10 dot 10x1 gives us 9x1
        u = np.dot(self.w2.T, h)
        # Run 1x9 through softmax to force each element to range of [0, 1] - 1x8
        y_c = self.softmax(u)
        return y_c, h, u


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backprop(self, e, h, x):
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.outer.html
        # Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
        # Going backwards, we need to take derivative of E with respect of w2
        # h - shape 10x1, e - shape 9x1, dl_dw2 - shape 10x9
        dl_dw2 = np.outer(h, e)
        # x - shape 1x8, w2 - 5x8, e.T - 8x1
        # x - 1x8, np.dot() - 5x1, dl_dw1 - 8x5
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        # Update weights
        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)


    # Get vector from word
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w


    # Input vector, returns nearest word(s)
    def vec_sim(self, word, top_n):
        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.v_count):
            # Find the similary score for each word in vocab
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)

# Initialise object
w2v = Word2Vec()
# Numpy ndarray with one-hot representation for [target_word, context_words]
training_data = w2v.generate_training_data(settings, corpus)
# Training
w2v.train(training_data)
# Get vector for word
vec = w2v.word_vec("machine")
# Find similar words
w2v.vec_sim("machine", 3)