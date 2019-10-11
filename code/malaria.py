#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:49:21 2019

@author: Andre
"""

import numpy  as np
import matplotlib.pyplot as plt
from scipy.special import expit
import cv2
import os

x = list() #create x data
y = list() # create y data
for i in os.listdir("../dataset/Parasitized"): #read all parasitized data 
    if ".png" in i: #this if block for only read .png files
        path = "../dataset/Parasitized/"+i # create path
        img = plt.imread(path) # and read created path
        img = cv2.resize(img,(40,40)) # resize image for lower processing power
        x.append(img) # append image to x data
        y.append(1) 
for i in os.listdir("../dataset/Uninfected"):
    if ".png" in i:
        path = "../dataset/Uninfected/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(40,40))
        x.append(img)
        y.append(0)
        
x = np.array(x) 
y = np.array(y)
#reshapeing data
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
y = y.reshape(y.shape[0], 1)
x = x**8

# i used sklearn modul for splitting process
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

# Plotting dataset images function
def plotDataset():
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(X[1],cv2.COLOR_BGR2RGB))
    plt.axis("on")


# Dataset normalization function
def normalize(image):
    image = image/image.max()
    return image

# Function for randomized creation of train, validation and test sets (60%, 20%, 20%)
def randomSetsCreation(X_norm, Y_norm):
    
    indices = np.random.permutation(X_norm.shape[0])
    train_idx = indices[:int(len(indices) * 0.8)]
    validation_idx = indices[len(train_idx):int(len(indices) * 0.8)]
    test_idx = indices[len(validation_idx) + len(train_idx):]

    X_train = X_norm[train_idx]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
    
    Y_train = Y_norm[train_idx]
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    
    X_validation = X_norm[validation_idx]
    X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1] * X_validation.shape[2] * X_validation.shape[3])
    
    Y_validation = Y_norm[validation_idx]
    
    X_test = X_norm[test_idx]
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3])
    
    Y_test = Y_norm[test_idx]
    
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test

# Sigmoid function
def sigmoid(x):
    return expit(x)

# Method for calculating the derivative of Sigmoid function
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# Leaky ReLU:
def leaky(x):
  return x*0.01 if x < 0 else x

# Leaky ReLU derivative:
def leaky_deriv(x):
  return 0.01 if x < 0 else 1

def calculate_accuracy(xdentro, ydentro, weights, bias, epoca):
    corretti = 0
    for i, image in enumerate(xdentro):
        result = sigmoid(np.dot(image, weights) + bias)
        if (float(result) > 0.5 and int(ydentro[i]) == 1):
            corretti += 1
        elif (float(result) < 0.5 and int(ydentro[i]) == 0):
            corretti += 1

    print("Epoca: " + str(epoca[-1]))
    print("Corretti: " + str(corretti))
    print("Accuracy: " + str(corretti/len(ydentro)))
    return float(corretti/len(ydentro))

#x = []
#
#with open("../dataset/sonar_x.txt") as f:
#    t = f.read()
#x = list(t.split())
#x = np.array(x, float)
#x = x.reshape(208, 60)
#x_train = np.array(x)
#with open("../dataset/sonar_y.txt") as g:
#    t = g.read()
#
#y = list(t.split())
#for i, value in enumerate(y):
#    if float(value) < 0:
#        y[i] = 0
#    else:
#        y[i] = 1
#y = np.array(y)
#y = y.reshape(y.shape[0], 1)
#
#
#X_norm = normalize(X)
#Y_norm = Y
#X_train, Y_train, X_validation, Y_validation, X_test, Y_test = randomSetsCreation(X_norm, Y_norm)


np.random.seed(42)
weights = np.random.randn(x_train.shape[1], 1)*np.sqrt(2/x_train.shape[1])
bias = np.random.rand(1)
lr = 0.05
#block = x_train.shape[0]
#x_test = x_test[180:]
#y_test = y_test[180:]
#x_train = x_train[:180]
#y_train = y_train[:180]





cost = []
test = []
epoca = []
print("Inizio:")
for i, epoch in enumerate(range(10000)):
    inputs = x_train

    # feedforward step1
    xw = np.dot(x_train, weights) + bias

    #feedforward step2
    z = sigmoid(xw)


    # backpropagation step 1
    error = z - y_train
    if(i % 100 == 0):
        epoca.append(i)
        xt = calculate_accuracy(x_train, y_train, weights, bias, epoca)
        xy = calculate_accuracy(x_test, y_test, weights, bias, epoca)
        cost.append(xt)
        test.append(xy)
        
        
    # backpropagation step 2
    dcost_dpred = error

    dpred_dz = sigmoid_der(z)


    z_delta = dcost_dpred * dpred_dz

    inputs = x_train.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num

plt.plot(epoca, cost)
plt.plot(epoca, test)
plt.show()
