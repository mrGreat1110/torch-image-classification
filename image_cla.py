import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import transformers
from torch.utils.data import DataLoader, Dataset
import torchvision 
import os
import glob
import pandas as pd
import cv2
torch.manual_seed(6)

class_names_label = {'buildings': 0,
                    'forest' : 1,
                    'glacier' : 2,
                    'mountain' : 3,
                    'sea' : 4,
                    'street' : 5
                    }
nb_classes = 6

path_train='C:/Users/nqh/Desktop/im_clas/venv/torch-image-classification/seg_train/seg_train'
path_pred='C:/Users/nqh/Desktop/im_clas/venv/torch-image-classification/seg_pred/seg_pred'
path_test='C:/Users/nqh/Desktop/im_clas/venv/torch-image-classification/seg_test/seg_test'
list_sub=os.listdir(path_train)
# create csv file which contain the class of image

id_image=[]
clas=[]
location=[]
for sub in list_sub:
    list_ima_class=os.listdir(path_train+'/'+sub)
    for ima in list_ima_class:
        id_image.append(ima)
        clas.append(sub)
        location.append(path_train+'/'+sub+'/'+ima)
df=pd.DataFrame({'image':id_image, 'class':clas, 'location':location})
df['class']=df['class'].replace(class_names_label)
df.to_csv('data_train.csv')

class Dataset(Dataset):
    def __init__(self, file_csv):
        super(Dataset, self).__init__()
        self.df=pd.read_csv(file_csv)
        self.name=self.df['image'].values
        self.clas=self.df['class'].values
        self.location=self.df['location'].values
    
    def __getitem__(self, index):
        image=cv2.imread(self.location[index])
        y=self.clas[index]
        return image, y


dataset=Dataset('data_train.csv')
print(dataset[3])
plt.imshow(dataset[3][0])
plt.show()
        

