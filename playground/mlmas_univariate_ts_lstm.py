# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

# Time Series Forecasting with the Long Short-Term Memory Network in Python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM


# ----------------------------------------------------------------------------------------------------------------
# Functions used in the subsequent application of the LSTM for forecasting
# ----------------------------------------------------------------------------------------------------------------
def cdate(x):
    """ Time parsing function for loading the dataset
     :param x: date string to be formatted
     :return: object of type datetime
     """
    return datetime.strptime('200' + str(x), '%Y-%m')


def timeseries_to_supervised(data, lag=1):
    """ Frame a sequence as a supervised learning problem
    :param data: time series as array
    :param lag: max lag to be considered among covariates
    :return: pandas data frame of format y,X
    """
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def difference(dataset, interval=1):
    """
    Create a differenced series.
    The same functionality could be achieved using Pandas shift() function
    :param dataset: original time series as pd data frame
    :param interval: order of difference, 1 by default
    :return: pd data frame of the differenced series
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


def inverse_difference(history, yhat, interval=1):
    """ Invert the differencing of a time series to obtain its original value"""
    return yhat + history[-interval]


def scale(train, test):
    """ Scale the training and test data to the interval [-1, 1] """
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def invert_scale(scaler, X, value):
    """ Inverse scaling for a forecasted value """
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def fit_lstm(train, nb_bs, nb_epoch, lstm_units):
    """
    The function takes care of setting up the LSTM and training it in stateful model.
    :param train: input series
    :param nb_bs: batch size to use for LSTM training
    :param nb_epoch: number of epochs to train the LSTM
    :param lstm_units: number of hidden units in the LSTM layer
    :return trained LSTM
    """
    # Reshaping to meet input requirements of the LSTM, which, for X are batch size, time steps, nb_features
    x, y = train[:, 0:-1], train[:, -1]
    x = x.reshape(x.shape[0], 1, x.shape[1])

    model = Sequential()
    model.add(LSTM(lstm_units,  # number of hidden units in the LSTM cell
                   batch_input_shape=(nb_bs, x.shape[1], x.shape[2]),  # dimension of the input data
                   stateful=True  # Crucial for time series forecasting
                   )
              )
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # LSTM training
    for i in range(nb_epoch):
        model.fit(x, y, epochs=1, batch_size=nb_bs, verbose=0, shuffle=False)  # no shuffling to sustain temporal order
        model.reset_states()
    return model


def forecast_lstm(model, obs, nb_bs=1):
    """
    One-step ahead forecast
    :param model: trained LSTM network
    :param obs: single observation for which we seek a predction
    :param nb_bs: batch_size; normally an argument but bound to be one in this example size we forecast 1-step ahead

    :return: LSTM prediction corresponding to obs
    """
    x = obs.reshape(1, 1, len(obs))
    yhat = model.predict(x, batch_size=nb_bs)
    return yhat[0, 0]


# ----------------------------------------------------------------------------------------------------------------
# Begin of the actual demonstration of univariate forecasting using LSTM
# ----------------------------------------------------------------------------------------------------------------
# Load the data
series = pd.read_csv('shampoo.csv', header=0, parse_dates=['Month'], index_col=0, squeeze=True,
                     date_parser=cdate)

print(series.head())
series.plot()
plt.show()

# transform the data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
print(pd.DataFrame(diff_values).head())
plt.plot(diff_values)
plt.show()

# transform differenced series to a supervised learning task
lag = 1
supervised = timeseries_to_supervised(diff_values, lag)
supervised_values = supervised.values

# split data into train and test-sets
nb_test_obs = 12  # take this many values at the end of the time series for hold-out testing
train, test = supervised_values[0:-nb_test_obs], supervised_values[-nb_test_obs:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the LSTM
neurons = 4
epochs = 1500
batch_size = 1
lstm_model = fit_lstm(train_scaled, batch_size, epochs, neurons)

# forecast the entire training dataset to build up state for forecasting the test set
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation using the hold-out test set
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]

    yhat = forecast_lstm(lstm_model, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

# report performance
rmse = np.sqrt(mean_squared_error(raw_values[-nb_test_obs:], predictions))
print('Test RMSE: {:.3f}'.format(rmse))
# line plot of observed vs predicted
plt.plot(raw_values[-nb_test_obs:], label='y')
plt.plot(predictions, ':', label='$\hat{y}$')
plt.legend(loc='best')
plt.show()

#
# # ----------------------------------------------------------------------------
# # Experimental test setup:
# # -------------------------------------------------------------
# # split data into train and test
# X = series.values
# train, test = X[0:-12], X[-12:]
#
# # walk-forward validation
# history = [x for x in train]
# predictions = list()
# for i in range(len(test)):
#     # make prediction
#     predictions.append(history[-1])
#     # observation
#     history.append(test[i])
# # report performance
# rmse = np.sqrt(mean_squared_error(test, predictions))
# print('RMSE: %.3f' % rmse)
# # line plot of observed vs predicted
# plt.plot(test)
# plt.plot(predictions)
# plt.show()
#
# # ----------------------------------------------------------------------------
# # LSTM data preparation
# # ----------------------------------------------------------------------------
#
# # test function to make supervised problem
# X = series.values
# supervised = timeseries_to_supervised(X, 1)
# print(supervised.head())
#
# # create a differenced series
# # ----------------------------------------------------------------------------
#
#
# # test function for differencing
# differenced = difference(series, 1)
# print(differenced.head())
#
# # invert transform
# inverted = list()
# for i in range(len(differenced)):
#     value = inverse_difference(series, differenced[i], len(series) - i)
#     inverted.append(value)
# inverted = pd.Series(inverted)
# print(inverted.head())
#
# # Scale the time series to [-1, 1]
# # ----------------------------------------------------------------------------
# X = series.values
# X = X.reshape(len(X), 1)
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler = scaler.fit(X)
# scaled_X = scaler.transform(X)
# scaled_series = pd.Series(scaled_X[:, 0])
# print(scaled_series.head())
# # invert transform
# inverted_X = scaler.inverse_transform(scaled_X)
# inverted_series = pd.Series(inverted_X[:, 0])
# print(inverted_series.head())
#
# # ----------------------------------------------------------------------------
# # LSTM Model Development
# # ----------------------------------------------------------------------------
# # Extract the data once more to facilitate re-starting the code from here
# # X = series.values
# train, test = X[0:-12].reshape(-1, 1), X[-12:].reshape(-1, 1)
# # train = train.reshape(-1, 1)
# # test = test
# # Define the LSTM
# hidden_neurons = 1
# # The batch_size must be set to 1. This is because it must be a factor of the size of the training and test data.
# # The predict() function of the model is also constrained by the batch size; there it must be set to 1 because we are
# # interested in making one-step forecasts on the test data.
# batch_size = 1
# epochs = 1500  # No tuning in this tutorial
