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
train_path =r'D:\STUDY\DS250\1b\fruits-360-original-size\fruits-360-original-size\Training'
list_labels = os.listdir(train_path)
test_paths = r'D:\STUDY\DS250\1b\fruits-360-original-size\fruits-360-original-size\Test'
test_labels = os.listdir(test_paths)


out = []
for item in range(0,7):
    item_path = test_paths +"\\"+ test_labels[item]
    test_img = os.listdir(item_path)
    correctness = 0
    total_count = 0
    for k in range(min(50,len(test_img))):
        img_array = cv2.imread(os.path.join(item_path,test_img[k]))
        img_array = cv2.resize(img_array,(128,64))
        fd,hog_image = hog(img_array,orientations=9, pixels_per_cell=(8,8),
                        cells_per_block=(2,2),visualize=True,channel_axis=-1)
        mini = []
        mini_name = []
        count =0 
        for label in list_labels:
            neighbor_path = train_path+"\\"+label
            train_imgs = os.listdir(neighbor_path)
            for i in range(0,len(train_imgs),5):
                train_array = cv2.imread(os.path.join(neighbor_path,train_imgs[i]))
                train_array =cv2.resize(train_array,(128,64))
                fd2,hog_image2 = hog(train_array,orientations=9, pixels_per_cell=(8,8),
                        cells_per_block=(2,2),visualize=True,channel_axis=-1)
                dist = np.sum((fd.astype("float") - fd2.astype("float")) ** 2)
	            #err /= float(prototype1.shape[0] * prototype1.shape[1])
                dist /= float(fd.shape[0])
                if count <= 2:
                    count+=1
                    mini.append(dist)
                    mini_name.append(label)
                else:
                    if mini[0] > dist:
                        mini.insert(0,dist)
                        mini.pop(3)
                        mini_name.insert(0,label)
                        mini_name.pop(3)
                    elif mini[1]>dist:
                        mini.insert(1,dist)
                        mini.pop(3)
                        mini_name.insert(1,label)
                        mini_name.pop(3)
                    elif mini[2]>dist:
                        mini.insert(2,dist)
                        mini.pop(3)
                        mini_name.insert(2,label)
                        mini_name.pop(3)
                
        
        if mini_name[0] == mini_name[1]:
            output = mini_name[0]
        elif mini_name[1] == mini_name[2]:
            output = mini_name[1]
        elif mini_name[0] == mini_name[2]:
            output = mini_name[0]
        else:
            output = mini_name[0]
        if output == item:
            correctness +=1 
        total_count+=1
    print((correctness,total_count))
    out.append([correctness,total_count])
sum1 =0
sum2 = 0 
for i in out:
    sum1 += i[0]
    sum2 += i[1] 
print(sum1)
print(sum2)
print("accuracy = ",sum1/sum2)

##################################
# after taking hog features and comparing them
# correctness = 897
# total count = 1197
# accuracy = 0.7493734335839599

