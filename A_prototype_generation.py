import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import torch



train_path =r'D:\STUDY\DS250\1b\fruits-360-original-size\fruits-360-original-size\Training'
list_labels = os.listdir(train_path)
#print(list_labels)
for item in list_labels:
    directory= r'D:\STUDY\DS250\prot_nodes'
    item_path = train_path +"\\"+ item
    count=0
    sum_img = np.asarray([0,0,0])
    #sum_img = sum_img.astype('int32')
    img_set = os.listdir(item_path)
    print(len(img_set))
    for img in img_set:
        img_array = cv2.imread(os.path.join(item_path,img))
        #rgb_img = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
        img_size = cv2.resize(img_array,(100,100))

        #sum_pixels +=rgb_img
        sum_img = sum_img + img_size


        count+=1
        
    avg_img = sum_img/(len(img_set))
    #plt.imshow(avg_img.astype('uint8'))
    #plt.show()
    os.chdir(directory)
    cv2.imwrite(item+".jpg",avg_img)
    #np.mean(sum_img, axis = (0))    
    
      
