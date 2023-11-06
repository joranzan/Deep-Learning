import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

import pickle
with open('C:/practice_data.txt', 'rb') as f:
    data = pickle.load(f)

overall = data.sum(axis=1)
print(overall)

max = np.max(data)
data_norm = data/max


def preprocess_rawdata(data, n=10):
    x = []
    y = []
    for i in range(data.shape[0]-n):
        last = i + n
        x.append(data[i:last]) # raw data를 sequencial하게 이어붙임
        y.append(data[last]) # x_train에 들어간 raw data의 다음 값을 target으로...
    print('input_shape:', np.array(x).shape)
    print('target_shape:', np.array(y).shape)
    return np.array(x), np.array(y)

x_data, y_data = preprocess_rawdata(data_norm)


def build_model():
    model = Sequential()
    model.add(SimpleRNN(52, activation='sigmoid'))
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

model = build_model()
h = model.fit(x_data, y_data, epochs=800, batch_size=20, verbose=0)
predict = model.predict(x_data)
predict = predict * max*52 # normalization 했으므로 다시 max곱하고 sum과 비교하기 위해 52개의 데이터 곱해줌
predict = predict.sum(axis=1)
predict_list = []
for i in range(x_data.shape[0]):
    if i<10:
        predict_list.append(0)
    else:
        predict_list.append(predict[i-10,])

predict = np.array(predict_list)
print(predict.shape)
print(overall.shape)
length = range(0, len(predict))
print(length)


def plot_graph():
    plt.plot(length, overall[length,])
    plt.plot(length, predict[length,])
    plt.title('COVID 19 of Overall Spain')
    plt.xlabel('Days')
    plt.ylabel('Confirmed Cases')
    plt.legend(['Overall','Predict'])

plot_graph()
plt.savefig("Q4.png")


