#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:19:36 2018

@author: kieling
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
import joblib
import os.path

def getSavedMODEL(fileName):
    print('Get saved MODEL')
    return joblib.load(fileName)

iris = datasets.load_iris()

X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.3)

redeNeural = MLPClassifier(max_iter=10000,tol=0.00001, learning_rate_init=0.01)

#loadModel
fileName = 'finalized_model.sav'

if os.path.isfile(fileName):    
    redeNeural = getSavedMODEL(fileName)
else:
    redeNeural.fit(X_train,y_train)

predict = redeNeural.predict(X_test)
joblib.dump(redeNeural,fileName)

print('[Error Squared]: ' + str(mean_squared_error(y_test,predict)))
