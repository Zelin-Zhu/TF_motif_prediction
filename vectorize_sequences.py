# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:47:19 2021

@author: ASUS
"""

#向量化编码序列

import pandas as pd
import numpy as np


class Vector_sequence:
    def __init__(self,file_path):
        self.file_path=file_path
        self.rawdata=pd.read_csv(self.file_path,header=None)
    def get_sequences(self):
        t=[]
        for i in range(int(self.rawdata.shape[0]/2)):
            t.append(self.rawdata[0][2*i+1])
        return t
    ##将序列编码并保存到file_name.npy文件中,file_name.npy
    def get_digital_sequences(self,file_name_dot_npy):
        fixed_sequences=self.get_sequences()
        temp=np.zeros((len(fixed_sequences),len(fixed_sequences[0]),4))
        for i in range(len(fixed_sequences)):
            for j in range(len(fixed_sequences[0])):
                if fixed_sequences[i][j]=='a' or fixed_sequences[i][j]=='A':
                    temp[i][j][0]=1
                if fixed_sequences[i][j]=='t' or fixed_sequences[i][j]=='T':
                    temp[i][j][1]=1
                if fixed_sequences[i][j]=='c' or fixed_sequences[i][j]=='C':
                    temp[i][j][2]=1
                if fixed_sequences[i][j]=='g' or fixed_sequences[i][j]=='G':
                    temp[i][j][3]=1
                if fixed_sequences[i][j]=='N' or fixed_sequences[i][j]=='n':
                    temp[i][j][0]=0.25 
                    temp[i][j][1]=0.25 
                    temp[i][j][2]=0.25 
                    temp[i][j][3]=0.25      
        np.save(file_name_dot_npy,temp)
        return temp
file_path='D:/workspace of spyder/毕业设计/my project/datafile/DNd41_background_seq.csv'
VS1=Vector_sequence(file_path)
df=VS1.get_sequences()
digital_seq=VS1.get_digital_sequences('temp.npy')

