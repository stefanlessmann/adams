
import pandas as pd
from numpy.random import choice
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout



N_train = 1000
x_train = np.zeros((N_train, 20))

one_indexes = choice(a=N_train, size=int(N_train / 2), replace=False)
x_train[one_indexes, 0] = 1  # very long term memory.
x_train = x_train.reshape(-1, 20, 1)
y_train = x_train[:, 0]


def prepare_sequences(x_train, y_train, window_length):
    windows = []
    windows_y = []
    for i, sequence in enumerate(x_train):
        len_seq = len(sequence)
        for window_start in range(0, len_seq - window_length + 1):
            window_end = window_start + window_length
            window = sequence[window_start:window_end]
            windows.append(window)
            windows_y.append(y_train[i])
    return np.array(windows), np.array(windows_y)


x_out, y_out = prepare_sequences(x_train, y_train, 10)

print('Building STATELESS model...')
model = Sequential()
model.add(LSTM(10, input_shape=(1,1, 1), return_sequences=False, stateful=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test), shuffle=False)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)