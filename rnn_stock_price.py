# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:40:56 2020

@author: ISSAM
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training set
name = r"C:\Users\ISSAM\Documents\GitHub\data\Google_Stock_Price_Train.csv"
dataset_train = pd.read_csv(name)
training_set = dataset_train.iloc[:,1:2]
training_set = training_set.values