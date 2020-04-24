import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from preview_mnist import get_images
from preview_mnist import get_labels

#start train model
print("TRAIN")
TRAINING_SIZE = 40000 #data (20k of 60k data)
train_images = get_images("mnist/train-images.idx3-ubyte", TRAINING_SIZE)
train_images = np.array(train_images)/255
train_labels = get_labels("mnist/train-labels.idx1-ubyte", TRAINING_SIZE)

#Support vector machine 
clf = svm.SVC(C=100)
clf.fit(train_images, train_labels)


#load data
TEST_SIZE = 2000 #test size data = 2000
test_images = get_images("mnist/t10k-images.idx3-ubyte", TEST_SIZE)  #get image function from preview_mnist.py
test_images = np.array(test_images)/255
test_labels = get_labels("mnist/t10k-labels.idx1-ubyte", TEST_SIZE) #get label function from preview_mnist.py
#end load data

#predict
print("PREDICT")
predict = clf.predict(test_images)
#end predict
#end train model

#result
print("RESULT")
ac_score = metrics.accuracy_score(test_labels, predict)
cl_report = metrics.classification_report(test_labels, predict)
print("Score = ", ac_score)
print(cl_report)
#end result

