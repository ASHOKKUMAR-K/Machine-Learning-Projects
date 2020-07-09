#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:24:11 2020

@author: ashokubuntu
"""


# Importing Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pickle

# Loading Data
iris =  pd.read_csv('data/Iris.csv')

# Extracting Features and prediction column
predictions = ['PetalLengthCm', 'PetalWidthCm', 'Species']

# Selecting Iris Features
iris_features = iris.loc[:, ['PetalLengthCm', 'PetalWidthCm']]
# Selecting Iris Species
iris_species = iris.loc[:, 'Species']

# Converting Pandas DataFrames and Series into Numpy arrays
features_values = iris_features.values
species_values = iris_species.values

# TODO : Split the dataset as Train set and Test set.
X_train, X_test, y_train, y_test = train_test_split(features_values, species_values, 
                                                    test_size = 0.25, random_state = 42)
                                                    
# TODO : Build Decision Tree Classifier Model
iris_classifier = DecisionTreeClassifier(criterion = 'gini')
iris_classifier.fit(X_train, y_train)

# Dumping Model to a pickle file
pickle.dump(iris_classifier, open("../models/iris_species_classifier.pkl", "wb"))