# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 18:21:14 2020

@author: ISSAM
"""
import tensorflow as tf
import theano
import numpy as np
#Preprocerssing Images
from keras.preprocessing.image import ImageDataGenerator
#step for CNN
from keras.models import Sequential
from keras.layers.convolutional import *
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.layers import Activation,Conv2D,MaxPool2D
from keras.metrics import categorical_crossentropy
#Data Normalisation
from keras.layers.normalization import BatchNormalization
#DataVisualisation
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import warnings
warnings.simplefilter(action="ignore",category=FutureWarning)

train_path = r"C:\Users\ISSAM\Desktop\Deep_Learning_A_Z\DL Colab Changes\Convolutional_Neural_Networks 3\dataset\training_set"
test_path =r"C:\Users\ISSAM\Desktop\Deep_Learning_A_Z\DL Colab Changes\Convolutional_Neural_Networks 3\dataset\test_set"

train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(256,256),classes = ['dogs','cats'],batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(256,256),classes = ['dogs','cats'],batch_size=4)

def plots(ims, figsize = (12,6),rows = 1,interp=False,titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1]!=3):
            ims = ims.transpose((0,3,2,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims)%2 == 2 else len(ims)//rows +1
    for i in range(len(ims)):
        sp =f.add_subplot(rows,cols,i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i],fontsize=16)
        plt.imshow(ims[i],interpolation = None if interp else 'none')

imgs,labels  = next(train_batches)
plots(imgs,titles=labels)

#initialise the models
model  = Sequential([
        Conv2D(filters=32,kernel_size=(3,3),activation = 'relu',padding = "same",input_shape=(256,256,3)),
        MaxPool2D(pool_size=(2,2),strides=2),
        Conv2D(filters=64,kernel_size=(3,3),activation = 'relu',padding = "same"),
        MaxPool2D(pool_size=(2,2),strides=2),
        Flatten(),
        Dense(units=2,activation='softmax'),
                    ])

model.summary()
model.compile(optimizer = Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=train_batches,validation_data = test_batches,epochs =10,verbose = 2)

model.save('cat_vs_dogs')
#Make call for the new model
new_model = tf.keras.models.load_model('cat_vs_dogs')
#making test for single images
from keras.preprocessing import image
#test1_path = r'C:\Users\ISSAM\Desktop\Deep_Learning_A_Z\DL Colab Changes\Convolutional_Neural_Networks 3\dataset\single_prediction\cat_or_dog_1.jpg'
#test_image = ImageDataGenerator().flow_from_directory(test1_path,target_size=(256,256))
path1 = r'C:\Users\ISSAM\Desktop\Deep_Learning_A_Z\DL Colab Changes\Convolutional_Neural_Networks 3\dataset\single_prediction\xx.jpg'
path2 = r'C:\Users\ISSAM\Desktop\Deep_Learning_A_Z\DL Colab Changes\Convolutional_Neural_Networks 3\dataset\single_prediction\cat_or_dog_2.jpg'
test_image = image.load_img(path1,target_size=(256,256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = new_model.predict(test_image)
y_pred_test_classes = np.argmax(result, axis=1)
y_pred_test_max_probas = np.max(result, axis=1)*100
test_batches.class_indices
print(result)


















        
    



