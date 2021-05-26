# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:38:05 2021

@author: ASUS
"""

import Binary_classification as bc
import numpy as np

def train_with_cellline(cell_line):
    
    ## 1.获取数据
    parent_file=''
    if cell_line=="DNd41":
        parent_file="DND41"
    else:
        parent_file=cell_line
        
    seqs_750_1000=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_750_1000.npy")
    seqs_500_750=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_500_750.npy")
    seqs_250_500=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_250_500.npy")
    seqs_0_250=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_0_250.npy")
    seqs_background=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_background.npy")
    seqs_background=seqs_background[0:8000]
    seqs_positive=seqs_0_250
    seqs_positive=np.append(seqs_positive,seqs_250_500,axis=0)
    seqs_positive=np.append(seqs_positive,seqs_500_750,axis=0)
    seqs_positive=np.append(seqs_positive,seqs_750_1000,axis=0)
      
    x_positive=seqs_positive
    y_positive=np.ones(len(x_positive))
    x_negtive=seqs_background
    y_negtive=np.zeros(len(x_negtive))
    y=np.append(y_positive,y_negtive,axis=0)
    x=np.append(x_positive,x_negtive,axis=0)
    
    ##对数据进行乱序
    index=[]
    for i in range(len(x)):
        index.append(i)
        
    np.random.shuffle(index)
    x=x[index]
    y=y[index]
    #3:1划分训练集合验证集
    num=int(len(x)*0.75)
    train_x=x[0:num]
    train_y=y[0:num]
    validate_x=x[num:-1]
    validate_y=y[num:-1]
    
    #训练模型
    BC_cellline=bc.Binary_classification(cell_line,train_x,train_y,validate_x,validate_y,"16000samples")
    BC_cellline.model_construct()
    BC_cellline.model_compile_and_fit()
    
    
    

train_with_cellline('DNd41')
train_with_cellline('GM12878')
train_with_cellline('H1hesc')
train_with_cellline('Helas3')

    
   