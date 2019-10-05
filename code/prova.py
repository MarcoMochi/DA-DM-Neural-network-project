import numpy  as np
import matplotlib.pyplot as plt
import cv2

X = np.load("../dataset/X.npy")
Y = np.load("../dataset/Y.npy")

# plotting dataset images
def plot_dataset():
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(X[1],cv2.COLOR_BGR2RGB))
    plt.axis("on")


#dataset normalization method
def normalize(image):
    image = image/image.max()
    return image

X_norm = normalize(X)

indices = np.random.permutation(X_norm.shape[0])
#print(indices)
train_idx = indices[:int(len(indices) * 0.6)]
validation_idx = indices[len(train_idx):int(len(indices) * 0.8)]
test_idx = indices[len(validation_idx) + len(train_idx):]

# randomized creation of train, validation and test sets (60%, 20%, 20%)
X_train = X_norm[train_idx]
Y_train = Y_norm[train_idx]

X_train = X_norm[validation_idx]
Y_train = Y_norm[validation_idx]

X_train = X_norm[test_idx]
Y_train = Y_norm[test_idx]