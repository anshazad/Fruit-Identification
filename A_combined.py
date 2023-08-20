#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import collections
import skimage
from skimage.io import imread , imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import torch


#start of code
test_path = r'D:\STUDY\DS250\1b\fruits-360-original-size\fruits-360-original-size\Test'
prototypes = r'D:\STUDY\DS250\prot_nodes'
directory= r'D:\STUDY\DS250\prot_hog'
list_labels = os.listdir(test_path)
train_path =r'D:\STUDY\DS250\1b\fruits-360-original-size\fruits-360-original-size\Training'
list_labels = os.listdir(train_path)


#variable declaration
correct = dict()

#data loading
for items in list_labels:
    correct[items] = 0
    
    item_path = test_path +"\\"+ items
    img_set = os.listdir(item_path)


    for img in img_set:
        img_array = cv2.imread(os.path.join(item_path,img))
        resize_img = resize(img_array,(128,64))
        compare = np.zeros(shape=(128,64))

        fd,hog_image = hog(resize_img,orientations=9, pixels_per_cell=(8,8),
                        cells_per_block=(2,2),visualize=True,channel_axis=-1)
        list_nodes = os.listdir(prototypes)
        t1=0
        min2 = np.zeros(shape=(128,64))
        count1 = 0
        count2 = 0
        count3 = 0
        t2 = 0
        t3 =0
        for i in range(len(list_nodes)):
            prototype = imread(os.path.join(prototypes,list_nodes[i]))
            prototype1 = resize(prototype,(128,64))
            fd2,hog_prot =hog(prototype1,orientations=9, pixels_per_cell=(8,8),
                        cells_per_block=(2,2),visualize=True,channel_axis=-1)
            err1 = np.sum((fd.astype("float") - fd2.astype("float")) ** 2)
            err1 /= float(fd.shape[0])
            err2 = np.sum((hog_image.astype("float") - hog_prot.astype("float")) ** 2)
            err2 /= float(hog_prot.shape[0] * hog_prot.shape[1])
            err3 = np.sum((resize_img.astype("float") - prototype1.astype("float")) ** 2)
	        #err /= float(prototype1.shape[0] * prototype1.shape[1])
            err3 /= float(prototype1.shape[0] * prototype1.shape[1])
            if count1 == 0 :
                count1 = err1
                #count+=1
            elif count1>err1:
                count1 = err1
                t1=i
            if count2 == 0 :
                count2 = err2
                #count+=1
            elif count2>err2:
                count2 = err2
                t2=i
            if count3 == 0 :
                count3 = err3
                #count+=1
            elif count3>err3:
                count3 = err3
                t3=i
        if list_nodes[t1][0:-4] == items or list_nodes[t2][0:-4] == items or list_nodes[t3][0:-4] == items:
            correct[items]+=1
    correct[items]=[correct[items],len(img_set)]
    print(correct[items])  
#algorithm
#testing step
length = 0
correctness = 0
for keys in correct:
    correctness += correct[keys][0]
    length +=correct[keys][1]     
accuracy = correctness/length
print(correctness)
print(length)
print(accuracy)
# #######################################################
# [157, 157]
# [101, 160]
# [132, 159]
# [132, 154]
# [99, 154]
# [84, 158]
# [160, 160]
# [228, 234]
# [121, 156]
# [79, 154]
# [124, 159]
# [40, 140]
# [150, 150]
# [79, 154]
# [156, 159]
# [47, 47]
# [50, 50]
# [50, 50]
# [49, 81]
# [77, 80]
# [115, 162]
# [72, 72]
# [80, 80]
# [66, 80]
# 2448
# 3110
# 0.7871382636655948  accuracy
