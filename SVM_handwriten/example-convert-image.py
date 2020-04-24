import cv2
import numpy as np
#get image
img = cv2.imread('image/nb_6.png',0)

#resize image
img = cv2.resize(img,(50,50))

#convert image to binary array
ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

#save to txt file
np.savetxt('array.txt', thresh_img,fmt='%4d')