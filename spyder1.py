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
from keras.layers import Activation
from keras.metrics import categorical_crossentropy
#Data Normalisation
from keras.layers.normalization import BatchNormalization
#DataVisualisation
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

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
        
    



