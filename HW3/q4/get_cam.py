from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
tf.compat.v1.disable_eager_execution()
K.set_learning_phase(0)

# load model
model = load_model('C:/Users/chohj/조한준/대학교/4학년 1학기/4학년 1학기/인공신경망과딥러닝/Homework/hw3/hw3_codes/qe3/1/pneumonia_q1.h5')

# image preprocessing
img_path = 'C:/Users/chohj/조한준/대학교/4학년 1학기/4학년 1학기/인공신경망과딥러닝/Homework/hw3/datasets/chest_xray/test/PNEUMONIA/person15_virus_46.jpeg'
img=image.load_img(img_path, target_size=(128,128))
img_tensor=image.img_to_array(img)
img_tensor=np.expand_dims(img_tensor,axis=0)
img_tensor=preprocess_input(img_tensor)

# gradCAM
def gradCAM(model, x):
    preds=model.predict(x)

    max_output=model.output[:,np.argmax(preds[0])]
    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(max_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0,1,2))

    iterate=K.function([model.input],[pooled_grads,last_conv_layer[0]])
    pooled_grads_value, conv_layer_output_value=iterate([x])
    for i in range(512):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
    heatmap=np.mean(conv_layer_output_value,axis=-1)
    heatmap=np.maximum(heatmap,0)
    heatmap/=np.max(heatmap)
    return heatmap


heatmap = gradCAM(model, img_tensor)
plt.matshow(heatmap)

#visualization
img=cv2.imread(img_path)
heatmap=cv2.resize(heatmap,(img.shape[1],img.shape[0]))
heatmap=np.uint8(255*heatmap)
heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img=heatmap*0.4+img
cv2.imwrite('/home/chohj0816/PycharmProjects/2017270187/hw3/q4/q4.png',superimposed_img)