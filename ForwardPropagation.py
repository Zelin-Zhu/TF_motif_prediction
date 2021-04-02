# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:20:01 2021

@author: ASUS
"""
import numpy as np
def seq_generator(seq_length,num):
    np.random.seed(12345)
    a=[[]]
    for i in range(num):
        temp=np.random.randint(0,4,seq_length)
        a.append(temp)
    a=a[1:]
    a=np.array(a)
    return a
seqs=seq_generator(120, 10)

def seq_coding(seqences,motif_lenght):
    S=[[[]]]
    edge=np.ones((motif_lenght-1,4))*0.25
    
    for i in range(seqences.shape[0]):
        seq_code=np.zeros((seqences.shape[1],4))
        for j in range(seqences.shape[1]):
            if seqences[i,j] is None:        ####当序列第i位置为none
                for k in range(4):
                    seq_code[j,k]=0.25
            else:
                seq_code[j,seqences[i,j]]=1
        seq_code=np.vstack((edge,seq_code))
        seq_code=np.vstack((seq_code,edge))
        S.append(seq_code)
    S=S[1:]
    S=np.array(S)
    return S
S=seq_coding(seqs, 4)

def convolution(input,motif_length,motif_num):
    M=np.random.random((motif_num,motif_length,4))
    X=[[]]
    for i in range(motif_num):
        x=np.zeros(input.shape[0]-motif_length+1)
        m=M[i].reshape(1,M[i].size)
        m=np.squeeze(m)            #将矩阵转化为一维向量
        for j in range(input.shape[0]-motif_length+1):
            temp=input[j:j+motif_length,:].reshape(1,M[i].size)
            temp=np.squeeze(temp)
            x[j]=np.dot(m,temp)
        X.append(x)    
    X=X[1:]
    X=np.array(X)
    return X

X=convolution(S[0], 4, 5)#五个卷积核，卷积核长度为4

def rectification(X,motif_num):
    Y=np.copy(X)
    B=np.random.random((1,motif_num))
    for i in range(motif_num):
        for j in range(X.shape[1]):
            if Y[i,j]<B[0,i]:
                Y[i,j]=0
            else:
                Y[i,j]=Y[i,j]-B[0,i]

    return Y

Y=rectification(X, 5)

def max_pooling(Y):
    Z=np.zeros((1,Y.shape[0]))
    for i in range(Y.shape[0]):
        Z[0,i]=max(Y[i])
    Z=np.squeeze(Z)
    return Z

Z=max_pooling(Y)

def Dence(input,input_length,output_length):
    #input_length=len(input)
    W=np.random.random((output_length,input_length))
    Bias=np.random.random((1,output_length))
    Out=np.zeros((1,output_length))
    W=np.matrix(W)
   
    input=input.reshape(input.size,1)
    input=np.matrix(input)
  
    Out=W*input
    Out=Out+Bias.transpose()
    return np.squeeze(Out)


Out=Dence(Z,5,4)
Out2=Dence(Out,4,1)


