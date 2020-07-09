#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 22:23:54 2020

@author: ashokubuntu
"""
# Importing Packages
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import pickle
# Loading Weight Height Data
wh = pd.read_csv('data/weight-height.csv')

# Extracting Prediction and Features values
X = wh.iloc[:, [1, 2]].values
y = wh.iloc[:, 0].values

# Splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                                    random_state = 42)

# Creating Support Vector Machine classifier Model
gender_classifier = make_pipeline(StandardScaler(), SVC(gamma='auto', probability = True))
gender_classifier.fit(X_train, y_train)

# Dumping Model to a pickle file
pickle.dump(gender_classifier, open("../models/gender_classifier.pkl", "wb"))