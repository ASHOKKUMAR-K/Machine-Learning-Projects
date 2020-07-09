#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 22:57:58 2020

@author: ashokubuntu
"""

# Importing Packages
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pickle
# Loading Weight Height Data
wh = pd.read_csv('data/weight-height.csv')

# Converting Categorical values to Numerical values
wh['Gender'] = wh['Gender'].apply(lambda x : {'Male' : 1, 'Female' : 0}[x])

# Extracting Prediction and Features values
X = wh.iloc[:, [1, 0]].values
y = wh.iloc[:, 2].values

# Splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                                    random_state = 42)

# Creating Support Vector Machine classifier Model
weight_predictor = make_pipeline(StandardScaler(), LinearRegression())
weight_predictor.fit(X_train, y_train)

# Dumping Model to a pickle file
pickle.dump(weight_predictor, open("../models/weight_predictor.pkl", "wb"))