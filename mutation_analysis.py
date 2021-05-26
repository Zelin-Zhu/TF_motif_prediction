# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:07:49 2021

@author: ASUS
"""
import numpy as np
import matplotlib.pyplot as plt

file="D:/workspace of spyder/毕业设计/my project data/datafile/predicted_motifs/"

def mutation_analysis(cell_line):
    file="D:/workspace of spyder/毕业设计/my project data/datafile/predicted_motifs/"
    mutation_prediction=np.load(file+cell_line+"motif_mutation_prediction.npy",allow_pickle=True)
    rate=np.zeros(len(mutation_prediction))
    for i in range(len(mutation_prediction)):
        rate[i]=np.sum(mutation_prediction[i])/(len(mutation_prediction[i])*9)
    
    ######################画图  
    name_list=['0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.8-0.9','0.9-1']
    num_list=[]
    p=0
    while p<0.8:
        
        num_list.append(np.count_nonzero((rate>p)&(rate<=p+0.1)))
        p=p+0.1
   
    plt.grid(True)
    plt.figure()
    
    
    fig=plt.bar(name_list,num_list,0.8)
 
    plt.ylabel('num')
    plt.xlabel('percent of 1')
    plt.title(cell_line+ '_percent of 1 in mutation matrix')
    plt.savefig(file+cell_line+'_mutation_analysis.jpg')
    return rate

def mutation_analysis_mode(cell_line):
    file="D:/workspace of spyder/毕业设计/my project data/datafile/predicted_motifs/"
    mutation_prediction=np.load(file+cell_line+"motif_mutation_prediction.npy",allow_pickle=True)
    one_row_percent=np.zeros(9)
    for i in range(len(mutation_prediction)):
        one_row_percent=one_row_percent+np.sum(mutation_prediction[i],axis=0)/9
    one_row_percent=one_row_percent/len(mutation_prediction)
     ######################画图  
    name_list=['1','2','3','4','5','6','7','8','9']
   
  
    plt.figure()
    
    
    fig=plt.bar(name_list,one_row_percent,0.8)
    plt.ylim((0, 1))
    plt.ylabel('percent')
    plt.xlabel('mutation modes')
    plt.title(cell_line+ ' percent of 1 in mutation matrix rows')
    plt.savefig(file+cell_line+'_mutation_analysis_row.jpg')
    
    
    return one_row_percent
        

rate_DNd41=mutation_analysis('DNd41')
rate_GM12878=mutation_analysis('GM12878')
rate_H1hesc=mutation_analysis('H1hesc')
rate_Helas3=mutation_analysis('Helas3')

row_percent_DNd41=mutation_analysis_mode('DNd41')
row_percent_GM12878=mutation_analysis_mode('GM12878')
row_percent_H1hesc=mutation_analysis_mode('H1hesc')
row_percent_Helas3=mutation_analysis_mode('Helas3')