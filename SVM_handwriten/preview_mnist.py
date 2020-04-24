import os
import numpy as np
from skimage import io

#convert data from file binary to array
#convert images
def get_images(img_file, number):
    f = open(img_file, "rb") # Open file in binary mode
    f.read(16) # Skip 16 bytes header
    images = []

    for i in range(number):
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    return images

#convert lables
def get_labels(label_file, number):
    l = open(label_file, "rb") # Open file in binary mode
    l.read(8) # Skip 8 bytes header
    labels = []
    for i in range(number):
        labels.append(ord(l.read(1)))
    return labels
#end convert data from file binary to array

#convert to png for preview
def convert_png(images, labels, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    for i in range(len(images)):
        out = os.path.join(directory, "%06d-num%d.png"%(i,labels[i]))
        io.imsave(out, np.array(images[i]).reshape(28,28))

number = 100 #data get from binary file
train_images = get_images("mnist/train-images.idx3-ubyte", number)
train_labels = get_labels("mnist/train-labels.idx1-ubyte", number)

convert_png(train_images, train_labels, "preview") #image save to preview folder
#end convert to png


#convert image to csv
def output_csv(images, labels, out_file):
    o = open(out_file, "w")
    for i in range(len(images)):
        o.write(",".join(str(x) for x in [labels[i]] + images[i]) + "\n")
    o.close()
#end convert image to csv