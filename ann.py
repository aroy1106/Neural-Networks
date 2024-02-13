# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 20:00:22 2024

@author: aroy2
"""
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:,3:13]    # [:,3:13] means first part before comma represents fetch all rows and 3:13 means extract 4th to second last column
y = dataset.iloc[:,13]

# creating dummy variables for categorical features
geography = pd.get_dummies(x['Geography'],drop_first=True)
gender = pd.get_dummies(x['Gender'],drop_first=True)


x = pd.concat([x,geography,gender],axis = 1)
x = x.drop(['Geography','Gender'],axis = 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# creating an empty ANN
classifier = Sequential()

# adding the input layer and the first hidden layer
classifier.add(Dense(units = 16, kernel_initializer = 'he_uniform', activation = 'leaky_relu', input_dim = 11))

# adding second hidden layer
classifier.add(Dense(units = 8 , kernel_initializer = 'he_uniform', activation = 'leaky_relu'))

# adding third hidden layer
classifier.add(Dense(units = 24 , kernel_initializer = 'he_uniform' , activation = 'leaky_relu'))

# adding the output layer
classifier.add(Dense(units = 1 , kernel_initializer = 'glorot_uniform' , activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

# fitting the ANN to the training set
model_history = classifier.fit(x_train , y_train , validation_split = 0.33 , batch_size = 10 , epochs = 100)

# predicting the test set
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculating the accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)