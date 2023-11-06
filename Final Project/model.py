from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history ['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def plot_loss(h, title="loss"):
    plt.plot(h.history ['loss'])
    plt.plot(h.history ['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)



train_path = 'trans/train'
test_path = 'trans/test'

height , width = 256 , 256

batch_size = 20

train_datagen = ImageDataGenerator(rescale=1./255, validation_split= 0.2, rotation_range=20, shear_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
                                   zoom_range=0.1, horizontal_flip= True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = (height, width),
    batch_size= batch_size,
    class_mode='binary',
    subset='training'
)

valid_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = (height, width),
    batch_size= batch_size,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = (height, width),
    batch_size= batch_size,
    class_mode='binary'
)
'''
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
'''

# model definition
input_shape = [height, width, 3] # as a shape of image
def build_model():
    model=models.Sequential()
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    conv_base.trainable=False
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model



# learning
import time
starttime=time.time();
num_epochs = 150
model = build_model()
history = model.fit_generator(train_generator, epochs=num_epochs, validation_data=valid_generator)

model.summary()

# evaluation
train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate_generator(test_generator)
print('train_acc:', train_acc)
print('train_loss : ', train_loss)
print('test_acc:', test_acc)
print('test_loss : ', test_loss)
print("elapsed time (in sec): ", time.time()-starttime)




model.save('non_fine_tune.h5')

plot_loss(history)
plt.savefig('loss.png')
plt.clf()
plot_acc(history)
plt.savefig('acc.png')
plt.clf()

########### fine tuning ################

base_model = model.layers[0]
base_model.trainable = True

for layer in base_model.layers :
    if layer.name[0:6] == 'block5' :
        layer.trainable = True
    else :
        layer.trainable = False

model.summary()
model.compile(optimizer=optimizers.RMSprop(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# learning
import time
starttime2 =time.time();
num_epochs = 100
model = build_model()
history2 = model.fit_generator(train_generator, epochs=num_epochs, validation_data=valid_generator)


# evaluation
train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate_generator(test_generator)
print('train_acc:', train_acc)
print('train_loss : ', train_loss)
print('test_acc:', test_acc)
print('test_loss : ', test_loss)

model.save('fine_tune.h5')


print("elapsed time (in sec): ", time.time()-starttime2)


plot_loss(history2)
plt.savefig('fine_tune_loss.png')
plt.clf()
plot_acc(history2)
plt.savefig('fine_tune_acc.png')
plt.clf()



