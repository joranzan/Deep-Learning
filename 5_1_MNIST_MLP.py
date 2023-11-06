from tensorflow.keras import models
from tensorflow.keras import layers

#ANN modeling 하는 함수 선언
def ANN_seq_func(Nin, Nh, Nout):
    model = models.Sequential()
    model.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)) )
    model.add(layers.Dense(Nout, activation='softmax') )
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',metrics=['accuracy'])
    return model

#Train_data: size(28*28) length(60000) label(0~9)
#Test_data : size(28*28) length(10000) label(0~9)
network = ANN_seq_func(28*28,512,10)

from tensorflow.keras.datasets import mnist
#data load
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#data reshape and normalization
L,W,H = train_images.shape
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

# binary target (one hot encoding)
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#fitting
h = network.fit(train_images, train_labels, epochs=20,
                batch_size=100, validation_split=0.2)

#evaluate on test set
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:',test_acc)

#plot learning curve
import matplotlib.pyplot as plt

def plot_acc(h,title="accuracy"):
  plt.plot(h.history['accuracy'])
  plt.title(title)
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')

plot_acc(h)