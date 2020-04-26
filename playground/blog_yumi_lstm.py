#https://fairyonice.github.io/Extract-weights-from-Keras's-LSTM-and-calcualte-hidden-and-cell-states.html

# Topic: LSTM, Keras, where to find LSTM parameter, time series forecasting

import matplotlib.pyplot as plt
from keras import models
from keras import layers

import seaborn as sns
import pandas as pd
import sys, time
import numpy as np

np.random.seed(123)
def random_sample(len_ts=3000, D=1001):
    c_range = range(5, 100)
    c1 = np.random.choice(c_range)
    u = np.random.random(1)
    const = -1.0 / len_ts
    ts = np.arange(0, len_ts)

    x1 = np.cos(ts / float(1.0 + c1))
    x1 = x1 * ts * u * const

    y1 = np.zeros(len_ts)

    for t in range(D, len_ts):
        ## the output time series depend on input as follows:
        y1[t] = x1[t - 2] * x1[t - D]
    y = np.array([y1]).T
    X = np.array([x1]).T
    return y, X


def generate_data(D=1001, Nsequence=1000, T=4000):
    X_train = []
    y_train = []

    for isequence in range(Nsequence):
        y, X = random_sample(T, D=D)
        X_train.append(X)
        y_train.append(y)
    return np.array(X_train), np.array(y_train)


D = 10
T = 1000
X, y = generate_data(D=D, T=T, Nsequence=1000)
print(X.shape, y.shape)

def plot_examples(X,y,ypreds=None,nm_ypreds=None):
    fig = plt.figure(figsize=(16,10))
    fig.subplots_adjust(hspace = 0.32,wspace = 0.15)
    count = 1
    n_ts = 16
    for irow in range(n_ts):
        ax = fig.add_subplot(n_ts/4,4,count)
        ax.set_ylim(-0.5,0.5)
        ax.plot(X[irow,:,0],"--",label="x1")
        ax.plot(y[irow,:,:],label="y",linewidth=3,alpha = 0.5)
        ax.set_title("{:}th time series sample".format(irow))
        if ypreds is not None:
            for ypred,nm in zip(ypreds,nm_ypreds):
                ax.plot(ypred[irow,:,:],label=nm)
        count += 1
    plt.legend()
    plt.show()
plot_examples(X,y,ypreds=None,nm_ypreds=None)


def define_model(len_ts,
                 hidden_neurons=1,
                 nfeature=1,
                 batch_size=None,
                 stateful=False):
    in_out_neurons = 1

    inp = layers.Input(batch_shape=(batch_size, len_ts, nfeature), name="input")

    rnn = layers.LSTM(hidden_neurons,
                      return_sequences=True,
                      stateful=stateful,
                      name="RNN")(inp)

    dens = layers.TimeDistributed(layers.Dense(in_out_neurons, name="dense"))(rnn)
    model = models.Model(inputs=[inp], outputs=[dens])

    model.compile(loss="mean_squared_error",
                  sample_weight_mode="temporal",
                  optimizer="rmsprop")
    return (model, (inp, rnn, dens))

X_train, y_train = X ,y
hunits = 3
model1, _ = define_model(hidden_neurons=hunits, len_ts=X_train.shape[1])
model1.summary()

# first values of y_t not defined based on x_t so do not compute MSE, which is achieved by setting weights to the loss
w = np.zeros(y_train.shape[:2])
w[:,D:] = 1
w_train = w

#Model training¶
from keras.callbacks import ModelCheckpoint
start = time.time()
hist1 = model1.fit(X_train, y_train,
                   batch_size=2**9,
                   epochs=200,
                   verbose=False,
                   sample_weight=w_train,
                   validation_split=0.05,
                   callbacks=[
                   ModelCheckpoint(filepath="weights{epoch:03d}.hdf5")])
end = time.time()
print("Time took {:3.1f} min".format((end-start)/60))

#The validation loss plot
labels = ["loss","val_loss"]
for lab in labels:
    plt.plot(hist1.history[lab],label=lab + " model1")
plt.yscale("log")
plt.legend()
plt.show()

# validate model with new data
X_test, y_test = generate_data(D=D,T=T,seed=2, Nsequence = 1000)
y_pred1 = model1.predict(X_test)

w_test = np.zeros(y_test.shape[:2])
w_test[:,D:] = 1
plot_examples(X_test,y_test,ypreds=[y_pred1],nm_ypreds=["ypred model1"])
print("The final validation loss is {:5.4f}".format(
    np.mean((y_pred1[w_test == 1] - y_test[w_test==1])**2 )))

#Reproduce LSTM layer outputs by hands
for layer in model1.layers:
        if "LSTM" in str(layer):
            weightLSTM = layer.get_weights()
warr,uarr, barr = weightLSTM
warr.shape,uarr.shape,barr.shape


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x)))


def LSTMlayer(weight, x_t, h_tm1, c_tm1):
    '''
    c_tm1 = np.array([0,0]).reshape(1,2)
    h_tm1 = np.array([0,0]).reshape(1,2)
    x_t   = np.array([1]).reshape(1,1)

    warr.shape = (nfeature,hunits*4)
    uarr.shape = (hunits,hunits*4)
    barr.shape = (hunits*4,)
    '''
    warr, uarr, barr = weight
    s_t = (x_t.dot(warr) + h_tm1.dot(uarr) + barr)
    hunit = uarr.shape[0]
    i = sigmoid(s_t[:, :hunit])
    f = sigmoid(s_t[:, 1 * hunit:2 * hunit])
    _c = np.tanh(s_t[:, 2 * hunit:3 * hunit])
    o = sigmoid(s_t[:, 3 * hunit:])
    c_t = i * _c + f * c_tm1
    h_t = o * np.tanh(c_t)
    return (h_t, c_t)


c_tm1 = np.array([0]*hunits).reshape(1,hunits)
h_tm1 = np.array([0]*hunits).reshape(1,hunits)



xs  = np.array([0.003,0.002,1])
for i in range(len(xs)):
    x_t = xs[i].reshape(1,1)
    h_tm1,c_tm1 = LSTMlayer(weightLSTM,x_t,h_tm1,c_tm1)
print("h3={}".format(h_tm1))
print("c3={}".format(c_tm1))

# We can calculate hidden states and cell states using Keras's functional API.
batch_size = 1
len_ts = len(xs)
nfeature = X_test.shape[2]

inp = layers.Input(batch_shape=(batch_size, len_ts, nfeature),
                   name="input")
rnn, s, c = layers.LSTM(hunits,
                        return_sequences=True,
                        stateful=False,
                        return_state=True,
                        name="RNN")(inp)
states = models.Model(inputs=[inp], outputs=[s, c, rnn])

for layer in states.layers:
    for layer1 in model1.layers:
        if layer.name == layer1.name:
            layer.set_weights(layer1.get_weights())

h_t_keras, c_t_keras, rnn = states.predict(xs.reshape(1, len_ts, 1))
print("h3={}".format(h_t_keras))
print("c3={}".format(c_t_keras))


fig = plt.figure(figsize=(9,4))
ax = fig.add_subplot(1,2,1)
ax.plot(h_tm1.flatten(),h_t_keras.flatten(),"p")
ax.set_xlabel("h by hand")
ax.set_ylabel("h by Keras")

ax = fig.add_subplot(1,2,2)
ax.plot(c_tm1.flatten(),c_t_keras.flatten(),"p")
ax.set_xlabel("h by hand")
ax.set_ylabel("h by Keras")
plt.show()