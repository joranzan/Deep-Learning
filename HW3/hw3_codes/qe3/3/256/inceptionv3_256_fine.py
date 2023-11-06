from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers

# set image generators
train_dir='C:/Users/chohj/조한준/대학교/4학년 1학기/4학년 1학기/인공신경망과딥러닝/Homework/hw3/datasets/chest_xray/train'
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,target_size=(256,256),batch_size=20,class_mode='binary'
)
validation_dir='C:/Users/chohj/조한준/대학교/4학년 1학기/4학년 1학기/인공신경망과딥러닝/Homework/hw3/datasets/chest_xray/val'
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,target_size=(256,256),batch_size=20,class_mode='binary'
)
test_dir='C:/Users/chohj/조한준/대학교/4학년 1학기/4학년 1학기/인공신경망과딥러닝/Homework/hw3/datasets/chest_xray/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,target_size=(256,256),batch_size=20,class_mode='binary'
)

# loading the model
model = load_model('pneumonia_q3_256.h5')

conv_base=model.layers[0]
for layer in conv_base.layers:
    if layer.name.startswith('block5'):
        layer.trainable = True

model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
              loss='binary_crossentropy', metrics=['accuracy'])

# main loop without cross_validation
import time
starttime=time.time()
num_epochs=100
history = model.fit_generator(train_generator,
                              epochs=num_epochs,
                              validation_data=validation_generator)

model.save('pneumonia_q3_256_fine.h5')

# evaluation

train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate_generator(test_generator)
print('train_acc:',train_acc)
print('train_loss:',train_loss)
print('test_acc:',test_acc)
print('test_loss:',test_loss)
print("elapsed time (in sec):",time.time()-starttime)

# visualization

def plot_acc(h, title="accuracy"):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training','Validation'],loc=0)

def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training','Validation'],loc=0)

plot_loss(history)
plt.savefig('Q3.256.after-FT.loss.png')
plt.clf()
plot_acc(history)
plt.savefig('Q3.256.after-FT.acc.png')
plt.clf()

# Scores
from sklearn.metrics import confusion_matrix,roc_auc_score
y_pred = model.predict_generator(test_generator)
matrix = confusion_matrix(test_generator.classes, y_pred>0.5)
auc = roc_auc_score(test_generator.classes, y_pred)
print('matrix:',matrix)
print('auc:', auc)