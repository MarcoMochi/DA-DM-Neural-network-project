#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:21:55 2019

@author: Andre
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
#np.random.seed(0)
#feature_set, labels = datasets.make_moons(100, noise=0.10)

#labels = labels.reshape(100, 1)

#x = np.load("../dataset/malaria_input.npy")
#y = np.load("../dataset/malaria_output.npy")

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
        
grey = []        
for i in range(len(x[:10])): 
    for k in range(40):
        for j in range(40):
            tot = x[i][k][j][0] * 0.3 + x[i][k][j][1] * 0.59 + x[i][k][j][2] * 0.11
            grey.append(tot)
        
x = np.array(grey) 
y = np.array(y)
#reshapeing data
x = x.reshape(10,1600)
y = y.reshape(y.shape[0], 1)
x = x**8
np.save("../dataset/malaria_input_grey", x)
np.save("../dataset/malaria_output_grey", y)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 50)



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

def sigmoid(x):
    return expit(x)

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

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


wh = np.random.rand(len(x_train[0]), 4)  
wo = np.random.rand(4, 1) 
lr = 25

x_test = x_test[:5000]
y_test = y_test[:5000]
x_train = x_train[:5000]
y_train = y_train[:5000]

cost = []
epoca = []

for i, epoch in enumerate(range(1000)):
    # feedforward
    zh = np.dot(x_train, wh)
    ah = sigmoid(zh)

    zo = np.dot(ah, wo)
    ao = sigmoid(zo)

    # Phase1 =======================

    error_out = ((1 / 2) * (np.power((ao - y_train), 2)))
    print(error_out.sum())       
    if (i % 100 == 0):
        corretti = 0
        epoca.append(i)
        for j, image in enumerate(ao):
            if image > 0.5 and y_train[j] == 1:
                corretti += 1
            elif image < 0.5 and y_train[j] == 0:
                corretti += 1
        cost.append(corretti/len(y_train))
        print(corretti/len(y_train))      

    dcost_dao = ao - y_train
    dao_dzo = sigmoid_der(zo) 
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

    # Phase 2 =======================

    # dcost_w1 = dcost_dah * dah_dzh * dzh_dw1
    # dcost_dah = dcost_dzo * dzo_dah
    dcost_dzo = dcost_dao * dao_dzo
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh) 
    dzh_dwh = x_train
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    # Update Weights ================

    wh -= lr * dcost_wh
    wo -= lr * dcost_wo
    
plt.plot(epoca, cost)
plt.show()