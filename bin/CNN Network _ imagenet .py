
# coding: utf-8

# In[4]:


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Dense, Activation, Flatten
from keras.layers import Input
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time
img_width, img_height = 224, 224
model = VGG16(weights = "imagenet" , include_top = True , input_shape=(img_width ,img_height ,3) )
img_path = "/home/Downloads/demo.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
#print  x.shape
result = model.predict(x)
#print model.summary()
img_input = Input(shape=(224, 224, 3))

#print decode_predictions(result)


# In[19]:


model = VGG16(input_tensor=img_input, include_top=True,weights='imagenet')
last_layer = model.get_layer("fc2").output
out = Dense(units = 4154, activation='softmax', name='output')(last_layer)
# print out 
custom_vgg_model = Model(img_input, out)

for i in custom_vgg_model.layers[:-1]:
    i.trainable = False

custom_vgg_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

training_set = train_datagen.flow_from_directory('/home/train_data',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/home/train_data',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

custom_vgg_model.fit_generator(training_set,
                         steps_per_epoch = 4154,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 4154
                        )


