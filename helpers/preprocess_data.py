from __future__ import division, print_function, absolute_import
import cv2
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import os
# Import MINST data
import pickle
from helpers import normalize, multiply_with_overflow, random_chunk1, random_chunk2


train = pickle.load(open('train.py', 'rb'))
# train = np.array([random_chunk2(p_image,100,32) for k in range(1000)])
# pickle.dump(train, open('train.py', 'wb'))


f = open('colourmap.txt','rb')
g = f.read().splitlines()
f.close()

C = np.array([[int(y) for y in x.split(' ')] for x in g])
cm = mpl.colors.ListedColormap(C/255.0)


image_name1 = '1073914843_B15_s2_w1.tif' #Red
image_name2 = '1073914843_B15_s2_w2.tif' #Green
path = './Run01test/'
image1 = cv2.imread(path + image_name1)
image2 = cv2.imread(path + image_name2)
R = np.array(image1[:,:,0])
G = np.array(image2[:,:,0])
B = np.zeros(R.shape)
raw_image = np.zeros([2160,2160,3])
raw_image[:,:,0], raw_image[:,:,1], raw_image[:,:,2] = R, G, B
n_image = normalize(raw_image)
p_image = multiply_with_overflow(n_image, [3,3,1])


    
