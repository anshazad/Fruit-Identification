import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import torch


test_path = r'D:\STUDY\DS250\1b\fruits-360-original-size\fruits-360-original-size\Test'
prototypes = r'D:\STUDY\DS250\prot_nodes'
list_labels = os.listdir(test_path)

correct = dict()
for items in list_labels:
    correct[items] = 0
    item_path = test_path +"\\"+ items
    img_set = os.listdir(item_path)


    for img in img_set:
        img_array = cv2.imread(os.path.join(item_path,img))
        img_size = cv2.resize(img_array,(100,100))
        compare = np.asarray([0,0,0])
        
        
        list_nodes = os.listdir(prototypes)
        #print(list_nodes[0][0:-4])
        t=0
        min2 = np.asarray([0,0,0])
        count = 0
        for i in range(len(list_nodes)):
            prototype = cv2.imread(os.path.join(prototypes,list_nodes[i]))
            prototype1 = cv2.resize(prototype,(100,100))
            compare = abs(img_size - prototype)
            #print(prototype)
            #plt.imshow(prototype.astype('uint8'))
            #plt.show()
            err = np.sum((img_size.astype("float") - prototype1.astype("float")) ** 2)
	        #err /= float(prototype1.shape[0] * prototype1.shape[1])
            err /= float(prototype1.shape[0] * prototype1.shape[1])
            #print(err)
            if count == 0 :
                count = err
                #count+=1
            elif count>err:
                count = err
                t=i
                #print(list_nodes[t][0:-4])
        if list_nodes[t][0:-4] == items:
            correct[items]+=1 
            #print(correct[items]) 
    correct[items]=[correct[items],len(img_set)] 
    print(correct[items])
length = 0
correctness = 0
for keys in correct:
    correctness += correct[keys][0]
    length +=correct[keys][1]     
accuracy = correctness/length
print(correctness)
print(length)
print(accuracy)
#######################################
#correctness = 2603
#length =3110
#accuracy = 0.8369774919614148