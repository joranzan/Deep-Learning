from tensorflow.keras import models, layers,optimizers


input_shape = [150, 150, 3]
# model definition
input_shape = [150, 150, 3] # as a shape of image
def build_model():
    model=models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
    input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
    loss='binary_crossentropy', metrics=['accuracy'])
    return model
