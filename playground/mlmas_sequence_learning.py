# https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/'
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM

# create sequence
length = 10
sequence = [i / float(length) for i in range(length)]
# create X/y pairs
df = pd.DataFrame(sequence)
df = pd.concat([df, df.shift(1)], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, y = values[:, 1], values[:, 0]
X = X.reshape(len(X), 1, 1)
print(X.shape, y.shape)
# configure network
n_batch = len(X)
n_epoch = 1000
n_neurons = 10
# design network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#fit the model
for i in range(n_epoch):
    model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
    model.reset_states()

# batch forecast
print('Batch forecast')
print('-------------------------------------------------------')
yhat = model.predict(X, batch_size=n_batch)
for i in range(len(y)):
	print('>Expected=%.1f, Predicted=%.1f' % (y[i], yhat[i]))


new_model = Sequential()
new_model.add(LSTM(n_neurons, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
new_model.add(Dense(1))
# copy weights
old_weights = model.get_weights()
new_model.set_weights(old_weights)
# compile model
new_model.compile(loss='mean_squared_error', optimizer='adam')

# online forecast
print('\nOnline forecast with new model')
print('-------------------------------------------------------')
for i in range(len(X)):
    testX, testy = X[i], y[i]
    testX = testX.reshape(1, 1, 1)
    yhat = new_model.predict(testX, batch_size=1)
    print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))
