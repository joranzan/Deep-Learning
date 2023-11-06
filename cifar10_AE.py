from tensorflow.keras import layers, models

class AE(models.Model):
    def __init__(self, x_nodes=3072, z_dim=36):
        x_shape = (x_nodes,)
        x = layers.Input(shape=x_shape)
        z = layers.Dense(z_dim, activation='relu')(x)
        y = layers.Dense(x_nodes, activation='sigmoid')(z)

        super().__init__(x,y)
        self.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        self.x = x
        self.z = z
        self.z_dim = z_dim

        def Encoder(self):
            return models.Model(self.x, self.z)

        def Decoder(self):
            z_shape = (self.z_dim,)
            z = layers.Input(shape=z_shape)
            y_layer = self.layers[-1]
            y = y_layer(z)
            return models.Model(z,y)


from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib as plt

def data_load():
    (x_train,_), (x_test, _) = cifar10.load_data()

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_train.shape[1:])))
    return (x_train, x_test)

def plot_acc(h, title="accuracy"):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

def main():
    x_nodes = 3072
    z_dim = 36

    (x_train, x_test) = data_load()
    autoencoder = AE(x_nodes, z_dim)

    history = autoencoder.fit(x_train, x_train,
                              epochs = 20,
                              batch_size=256,
                              shuffle=True,
                              validation_data=(x_test, x_test))

    plot_loss(history)
    plt.savefig('ae_cifar10.loss.png')
    plt.clf
    plot_acc(history)
    plt.savefig('ae_cifar10.acc.png')

    show_ae('ae_mnist.predicted.png')
    plt.show()

if __name__== '__main__':
    main()

def show_ae(autoencoder, x_test):
    encoder = autoencoder.Encoder()
    decoder = autoencoder.Decoder()

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10
    plt.figure(figsize=(20,6))
    for i in range(n):
        ax = plt.subplot(3, n, i+1)
        plt.imshow(x_test[i].reshape(32,32,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.stem(encoded_imgs[i].reshape(-1), use_line_collection=True)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(x_test[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

