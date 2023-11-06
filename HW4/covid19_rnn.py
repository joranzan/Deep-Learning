import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, GRU, LSTM
import matplotlib.pyplot as plt

import pickle
with open('C:/practice_data.txt', 'rb') as f:
    data = pickle.load(f)

print(data.shape)

train_data = data[:350]
test_data = data[350:]

print(train_data.shape)
print(test_data.shape)


def normalize(train_data, test_data):
    import copy
    train_data_norm = copy.deepcopy(train_data)
    test_data_norm = copy.deepcopy(test_data)
    max = train_data_norm.max(axis=0)
    train_data_norm /= max
    test_data_norm /= max
    print(train_data_norm)
    print(test_data_norm)
    return train_data_norm, test_data_norm

train_data_norm, test_data_norm = normalize(train_data, test_data)

def preprocess_rawdata(data, n=10):
    x = []
    y = []
    for i in range(data.shape[0]-n):
        last = i + n
        x.append(data[i:last]) # raw data를 sequential하게 이어붙임
        y.append(data[last]) # x_train에 들어간 raw data의 다음 값을 target으로...
    print('input_shape:', np.array(x).shape)
    print('target_shape:', np.array(y).shape)
    return np.array(x), np.array(y)

x_train, y_train = preprocess_rawdata(train_data_norm)
x_test, y_test = preprocess_rawdata(test_data_norm)

print(x_train)
print(x_test)

def build_model():
    model = Sequential()
    model.add(LSTM(52, activation='sigmoid'))
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss='mse')
    return model

def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training','Validation'])

# time series cross validation
tscv = TimeSeriesSplit(n_splits=4)
overall_loss = []
validation_loss = []
for train_index, val_index in tscv.split(x_train):
    model = build_model()
    h = model.fit(x_train[train_index], y_train[train_index],
                    validation_data=(x_train[val_index], y_train[val_index]),
                    epochs=1000, batch_size=20, verbose=0)
    validation_loss.append(h.history['val_loss'][-1])
    print('Loss', h.history['loss'][-1], '\nVal_Loss:', h.history['val_loss'][-1])
    plot_loss(h)
    plt.show()
    plt.savefig('loss.png')
    plt.clf()
validation_loss = np.array(validation_loss)
print(np.mean(validation_loss))


