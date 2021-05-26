# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:28:46 2021

@author: ASUS
"""

#models with different detector length
import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np
import Binary_classification as bc

parent_file="DND41"
cell_line="DNd41"

 #导入数据
seqs_750_1000=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_750_1000.npy")
seqs_500_750=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_500_750.npy")
seqs_250_500=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_250_500.npy")
seqs_0_250=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_0_250.npy")
seqs_background=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_background.npy")
seqs_background=seqs_background[0:1000]
seqs_positive=seqs_0_250[0:250]
seqs_positive=np.append(seqs_positive,seqs_250_500[0:250],axis=0)
seqs_positive=np.append(seqs_positive,seqs_500_750[0:250],axis=0)
seqs_positive=np.append(seqs_positive,seqs_750_1000[0:250],axis=0)

x_positive=seqs_positive
y_positive=np.ones(len(x_positive))
x_negtive=seqs_background
y_negtive=np.zeros(len(x_negtive))
y=np.append(y_positive,y_negtive,axis=0)
x=np.append(x_positive,x_negtive,axis=0)

index=[]
for i in range(len(x)):
    index.append(i)
        
np.random.shuffle(index)
x=x[index]
y=y[index]
num=1500

##构造数据集
train_x=x[0:num]
train_y=y[0:num]
validate_x=x[num:-1]
validate_y=y[num:-1]


for i in range(5,17):
    
  
    BC=bc.Binary_classification('DNd41_test'+str(i),train_x,train_y,validate_x,validate_y,'testdata')
    BC.model_construct(detector_length=i)
    BC.model_compile_and_fit()
   
    
   
#画图
import matplotlib.pyplot as plt
name_list=[]
num_list=[0.7,0.8,0.8,0.85,0.84,0.85,0.88,0.88,0.87,0.89,0.88]
for i in range(5,16):
    name_list.append(str(i))
    
plt.figure()   

   
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.25, 1.01*height, '%.2f' % height)
    
fig=plt.bar(range(len(num_list)), num_list,tick_label=name_list)  
autolabel(fig)
plt.ylabel('best validation accuracy')
plt.xlabel('detector length')
plt.title('best model validation accuracy with different detector lengths')
file='D:/workspace of spyder/毕业设计/my project data/model_file/'
plt.savefig(file+'best model validation accuracy with different detector lengths.jpg')