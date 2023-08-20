import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
import os
import tensorflow as tf
from tensorflow import keras
import cv2
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#checking for gpu
train_path =r'D:\STUDY\DS250\1b\fruits-360_dataset\fruits-360\Training'
train_labels = os.listdir(train_path)
test_path = r'D:\STUDY\DS250\1b\fruits-360_dataset\fruits-360\Test'
test_labels = os.listdir(test_path)
train_imgs =[]
train_imgs_name = []
test_imgs = []
test_imgs_name = []
index =0
index1 = 0
for item in train_labels:
    item_path = test_path +"\\"+ item
    item2_path = train_path + "\\"+item
    test_img_fold = os.listdir(item_path)
    train_img_fold = os.listdir(item2_path)
    for img in test_img_fold:
        img_array= cv2.imread(os.path.join(item_path,img))
        test_imgs.append(np.ravel(cv2.resize(img_array,(50,50))))
        test_imgs_name.append(index)
    for img in train_img_fold:
        img2_array = cv2.imread(os.path.join(item2_path,img))
        train_imgs.append(np.ravel(cv2.resize(img2_array,(50,50))))
        train_imgs_name.append(index1)
    index +=1
    index1+=1
#trainloader = torch.utils.data.DataLoader(train_imgs.astype(np.float32), batch_size=64, shuffle=True)
#testloader = torch.utils.data.DataLoader(test_imgs.astype(np.float32), batch_size=64, shuffle=True)
test_imgs = np.array(test_imgs)
train_imgs = np.array(train_imgs)
print(test_imgs.shape)
print(train_imgs.shape)
test_imgs_name = to_categorical(np.array(test_imgs_name))
train_imgs_name = to_categorical(np.array(train_imgs_name))
print(test_imgs_name)
print(test_imgs)




desc=len(train_imgs[0])
# model_ann = keras.Sequential([keras.layers.Dense(12,input_shape=(7500,),activation='relu'), # Use appropriate values 
                        
#                          keras.layers.Dense(12,activation='relu'),
                      
#                          keras.layers.Dense(131,activation='relu'),
#                          keras.layers.Dense(131,activation='softmax')]) # 131 is no. of dimensions in the output layer
model_ann = Sequential()
model_ann.add(Dense(800, input_shape=(7500,), activation='relu'))
model_ann.add(Dense(600, activation='relu'))
model_ann.add(Dense(400, activation="relu"))
model_ann.add(Dense(200, activation="relu"))
model_ann.add(Dense(131, activation='softmax'))
# Use the appropriate optimizer (gradient descent), loss function and metrics for evaluating classification accuracy
model_ann.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


model_ann.fit(train_imgs,train_imgs_name, epochs=10,batch_size = 64,validation_data = (test_imgs,test_imgs_name))
#model_ann.evaluate(test_imgs,test_imgs_name)

