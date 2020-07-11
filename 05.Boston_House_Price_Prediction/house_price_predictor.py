#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:49:51 2020

@author: ashokubuntu
"""

# Importing Packages
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import pickle

# Loading Data
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston = pd.read_csv('data/housing.csv', delimiter=r"\s+", 
                     names = columns)

# Selecting Top 6 Features
top_6_features = ['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'TAX', 'NOX']

# Extracting top features values
features = boston.loc[:, top_6_features].values

# Extracting Prediction values
predictions = boston.loc[:, 'MEDV'].values

# Splitting Data into train and test
# Choosing Optimal Training Samples
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    predictions,
                                                    test_size = 0.20,
                                                    random_state = 3)

# Building Optimal Random Forest regressor Model
random_forest_regressor = RandomForestRegressor(max_depth = 13,
                                                random_state = 68)
random_forest_regressor.fit(X_train, y_train)

# Dumping Model to a pickle file
pickle.dump(random_forest_regressor, open("../models/boston_house_price_predictor.pkl", "wb"))