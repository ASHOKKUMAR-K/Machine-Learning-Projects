#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 07:36:14 2020

@author: ashokubuntu
"""

# Importing Packages
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import pickle


# TODO : Filling Above values on LoanAmount column based on LoanStatus.
def LoanAmount_null_values_filler(df):
    ''' Fills null value on LoanAmount based on Loan_Status '''
    for row in range(df.shape[0]):
        if pd.isnull(df.loc[row, 'LoanAmount']):
            if df.loc[row, 'Loan_Status'] == 'Y':
                df.loc[row, 'LoanAmount'] = 151.22
            elif df.loc[row, 'Loan_Status'] == 'N':
                df.loc[row, 'LoanAmount'] = 144.29
            else:
                pass
    return df

# TODO : Function for filling null values on dependents columns
def dependents_null_value_filler(df):
    for row in range(df.shape[0]):
        if df.loc[row, 'Dependents'] is np.nan:
            df.loc[row, 'Dependents'] = str(df.loc[row, 'Married'])
    return df

# Loading Train data
train_data = pd.read_csv('data/train_u6lujuX_CVtuZ9i.csv')

# Initialzing Prediction features
prediction_features = ['Credit_History', 'LoanAmount', 'ApplicantIncome', 'CoapplicantIncome', 'Dependents', 'Loan_Status', 'Married']

# Extracting prediction data from the whole data
train_data = train_data.loc[:, prediction_features]

# TODO : Filling null values on Credit_History
train_data['Credit_History'] = train_data['Credit_History'].fillna(value = 1.0)

# TODO : Filling null values on LoanAmount
train_data = LoanAmount_null_values_filler(train_data)

# Let's fill null values in Married columns with 'Yes'
train_data['Married'] = train_data['Married'].fillna('Yes')

# TODO : encoding categorical values into numerical values
train_data['Married'] = train_data['Married'].apply(lambda x : {'Yes' : 1, 'No' : 0}[x])

# TODO : Fill null values on Dependents column
train_data = dependents_null_value_filler(train_data)

# TODO : Encoding Categorical data into Numerical Data
train_data['Dependents'] = train_data['Dependents'].apply(lambda x : {'0' : 0, '1' : 1, '2' : 2, '3+' : 3}[x])

# Drop Married Column
train_data.drop('Married', inplace = True, axis = 1)

# Extracting Feature Values and Prediction Values
feature_values = train_data.iloc[:, :-1].values
prediction_values = train_data['Loan_Status'].values

# Creating Decision Tree Classifier Model
decision_tree_model = DecisionTreeClassifier(max_depth = 8)
decision_tree_model.fit(feature_values, prediction_values)

# Dumping Model to a pickle file
pickle.dump(decision_tree_model, open("../models/loan_status_predictor.pkl", "wb"))