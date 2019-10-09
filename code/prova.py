import numpy  as np
import matplotlib.pyplot as plt
from scipy.special import expit
import cv2

X = np.load("../dataset/X.npy")
Y = np.load("../dataset/Y.npy")

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

def calculate_accuracy(X_test, weights, bias, epoca):
    corretti = 0
    for i, image in enumerate(X_test):
        result = sigmoid(np.dot(image, weights) + bias)
        if (float(result) > 0.5 and int(y[i]) == 1):
            corretti += 1
        elif (float(result) < 0.5 and int(y[i]) == 0):
            corretti += 1

    print("Epoca: " + str(epoca[-1]))
    print("Corretti: " + str(corretti))
    print("Accuracy: " + str(corretti/len(y)))
    return float(corretti/len(y))

x = []

with open("../dataset/sonar_x.txt") as f:
    for line in f.read():
        for value in line.split():
            x.append(value)
print(x)
y = []

with open("../dataset/sonar_y.txt") as g:
    for line in g.read():

        if(float(value) == -1):
            y.append(0)
        else:
            y.append(value)
            
x = np.array(x)
x = x.reshape(208, 60)

y = np.array(y)
y = y.reshape(208, 1)
            
X_norm = normalize(X)
Y_norm = Y
X_train, Y_train, X_validation, Y_validation, X_test, Y_test = randomSetsCreation(X_norm, Y_norm)

np.random.seed(42)
weights = np.random.randn(x.shape[1], 1)*np.sqrt(2/x.shape[1])
bias = np.random.rand(1)
lr = 0.05

cost = []
epoca = []
for i, epoch in enumerate(range(10000)):
    inputs = x

    # feedforward step1
    XW = np.dot(x, weights) + bias

    #feedforward step2
    z = sigmoid(XW)

    # backpropagation step 1
    error = z - y
    if(i % 100 == 0):
        epoca.append(i)
        xt = calculate_accuracy(x, weights, bias, epoca)
        cost.append(xt)
        
        
    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = x.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num




plt.plot(epoca, cost)
plt.show()
