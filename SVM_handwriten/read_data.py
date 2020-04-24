import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from preview_mnist import get_images
from preview_mnist import get_labels

#start train model
print("TRAIN")
TRAINING_SIZE = 1 #data (20k of 60k data)
train_images = get_images("mnist/train-images.idx3-ubyte", TRAINING_SIZE)
train_images = np.array(train_images)
train_labels = get_labels("mnist/train-labels.idx1-ubyte", TRAINING_SIZE)
np.savetxt('array1.txt', train_images,fmt='%4.5f')
print("Label: ",train_labels)
print(train_images)
