# -*- coding: utf-8 -*-
"""
Created on Sun May  2 13:38:02 2021

@author: ASUS
"""

#motif分析
import numpy as np
import matplotlib.pyplot as plt
#获取细胞系预测出motif的序列
def get_motif_seq(cell_line):
    parent_file=''
    if cell_line=="DNd41":
        parent_file="DND41"
    else:
        parent_file=cell_line
    seqs_750_1000=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+parent_file+"/"+cell_line+"_750_1000.npy")
    motifs=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/predicted_motifs/"+cell_line+"_motifs.npy")
    motif_seqs=[]
    for i in range (len(motifs)):
        left=motifs[i][0]
        right=motifs[i][1]
        num=motifs[i][2]
        motif_seq=''
        for j in range(left,right+1):
            if seqs_750_1000[num][j][0]==1:
                motif_seq=motif_seq+'A'
            elif seqs_750_1000[num][j][1]==1:
                motif_seq=motif_seq+'T'
            elif seqs_750_1000[num][j][2]==1:
                motif_seq=motif_seq+'C'
            elif seqs_750_1000[num][j][3]==1:
                motif_seq=motif_seq+'G'
            else:
                motif_seq=motif_seq+'N'
        motif_seqs.append(motif_seq)
    file="D:/workspace of spyder/毕业设计/my project data/datafile/predicted_motifs/"
    np.save(file+cell_line+"_"+"motif_seqs"+".npy",motif_seqs) 
                
    return motif_seqs

def motif_length_analysis(cell_line):
    file="D:/workspace of spyder/毕业设计/my project data/datafile/predicted_motifs/"
    motif_seqs=np.load(file+cell_line+"_"+"motif_seqs.npy")
    
    #找到最短的motif长度记录所有motif长度
    min_len=100
    length=np.zeros(len(motif_seqs))
    for i in range(len(motif_seqs)):
        length[i]=len(motif_seqs[i])
        if len(motif_seqs[i])<min_len:
            min_len=len(motif_seqs[i])   
    #从min_len开始到20每隔两个bp统计频率
    name_list=[]
    num_list=[]
    while min_len<20:
        name_list.append(str(min_len)+"-"+str(min_len+1))
        num_list.append(np.count_nonzero((length< min_len+2)&(length>=min_len)))
        min_len=min_len+2
    name_list.append(">="+str(min_len))
    num_list.append(np.count_nonzero(length>=min_len))
   
    plt.grid(True)
    plt.figure()
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2.-0.25, 1.01*height, '%d' % height)
    
    fig=plt.bar(range(len(num_list)), num_list,tick_label=name_list)  
    autolabel(fig)
    plt.ylabel('num')
    plt.xlabel('length')
    plt.title(cell_line+' prediced motif length distribution')
    
    
    file="D:/workspace of spyder/毕业设计/my project data/datafile/predicted_motifs/"
    plt.savefig(file+cell_line+'motif_length_distribution.jpg')
    
    return 0
  
    
    
DNd41_motif_seqs=get_motif_seq("DNd41")
GM12878_motif_seqs=get_motif_seq("GM12878")
H1hesc_motif_seqs=get_motif_seq("H1hesc")
Helas3_motif_seqs=get_motif_seq("Helas3")


motif_length_analysis('DNd41')
motif_length_analysis('GM12878')
motif_length_analysis('H1hesc')
motif_length_analysis("Helas3")