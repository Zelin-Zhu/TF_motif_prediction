# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:13:41 2021

@author: ASUS
"""
#低强度序列单独建立模型

import Binary_classification as bc
import numpy as np

def train_with_low_strength_seqs(cell_line):
    #导入数据
    parent_file=''
    if cell_line=="DNd41":
        parent_file="DND41"
    else:
        parent_file=cell_line
    seqs_0_250=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_0_250.npy")
    seqs_background=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_background.npy")
    x=seqs_0_250
    x=np.append(x,seqs_background[0:len(x)],axis=0)
    y_positive=np.ones((len(seqs_0_250),1))
    y_negetive=np.zeros((len(seqs_0_250),1))
    y=np.append(y_positive,y_negetive,axis=0)
    #乱序
    index=[]
    for i in range(len(x)):
        index.append(i)
        
    np.random.shuffle(index)
    x=x[index]
    y=y[index]
    
    train_x=x[0:int(len(x)*0.75)]
    train_y=y[0:int(len(x)*0.75)]
    validate_x=x[int(len(x)*0.75):-1]
    validate_y=y[int(len(x)*0.75):-1]
    data_name="low_strength_seqs"
    BC_cellline=bc.Binary_classification(cell_line,train_x,train_y,validate_x,validate_y,data_name)
    BC_cellline.model_construct()
    BC_cellline.model_compile_and_fit()
    
    
train_with_low_strength_seqs("DNd41")
train_with_low_strength_seqs("GM12878")
train_with_low_strength_seqs("Helas3")
train_with_low_strength_seqs("H1hesc")
        
    
