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



