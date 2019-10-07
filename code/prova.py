import numpy  as np
import matplotlib.pyplot as plt
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
    train_idx = indices[:int(len(indices) * 0.6)]
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
    return 1/(1+np.exp(-x))

# Method for calculating the derivative of Sigmoid function
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

def calculate_accuracy(X_test, weights, bias, epoca):
    corretti = 0
    for i, image in enumerate(X_test):
        result = sigmoid(np.dot(image, weights) + bias)
        if (float(result) > 0.5 and int(Y_test[i]) == 1):
            corretti += 1
        elif (float(result) < 0.5 and int(Y_test[i]) == 0):
            corretti += 1

    print("Epoca: " + str(epoca))
    print("Corretti: " + str(corretti))
    print("Accuracy: " + str(corretti/len(Y_test)))

X_norm = normalize(X)
Y_norm = Y
X_train, Y_train, X_validation, Y_validation, X_test, Y_test = randomSetsCreation(X_norm, Y_norm)

np.random.seed(42)
weights = np.random.rand(X_train.shape[1], 1)
bias = np.random.rand(1)
lr = 0.05

cost = []
epoca = []
for i, epoch in enumerate(range(101)):
    inputs = X_train

    # feedforward step1
    XW = np.dot(X_train, weights) + bias

    #feedforward step2
    z = sigmoid(XW)

    # backpropagation step 1
    error = z - Y_train
    if(i % 1 == 0):
        cost.append(int((0.5*(error.sum()*error.sum()))/X_train.shape[0]))
        epoca.append(i)
        #calculate_accuracy(X_test, weights, bias, epoca)
    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = X_train.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num




plt.plot(epoca, cost)
plt.show()
