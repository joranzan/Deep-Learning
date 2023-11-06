import pickle
import numpy as np

with open('practice_data.txt', 'rb') as f:
    data = pickle.load(f)

print(data.shape)

def generator(self, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=10):
    if max_index is None:
        max_index = len(self.data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i+batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback//step, self.data_length))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(row[j]-lookback, rows[j],step)
            samples[j] = self.data[indices]
            targets[j] = self.data[rows[j]+delay-1][1]
        yield samples, targets



