# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:08:12 2021

@author: ASUS
"""

##给出各细胞训练出的模型在不同强度序列上的预测准确度（positive_sample_predictions_accuracy_with_strength)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

model_DNd41=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/DNd41_16000samples_log/weights.38-0.93.hdf5")
model_GM12878=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/GM12878_16000samples_log/weights.14-0.91.hdf5")
model_H1hesc=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/H1hesc_16000samples_log/weights.24-0.93.hdf5")
model_Helas3=tf.keras.models.load_model("D:/workspace of spyder/毕业设计/my project data/model_file/Helas3_16000samples_log/weights.33-0.89.hdf5")
def predict_with_strength(model,cell_line):
     
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
    
    y_positive=np.ones((2000,1))
    acc_750_1000=model.evaluate(seqs_750_1000,y_positive)
    acc_500_750=model.evaluate(seqs_500_750,y_positive)
    acc_250_500=model.evaluate(seqs_250_500,y_positive)
    acc_0_250=model.evaluate(seqs_0_250,y_positive)
    acc_background=model.evaluate(seqs_background[0:2000],np.zeros((2000,1)))
    
    name_list = ['750-1000','500_750','250_500','0_250','background']  
    num_list = [acc_750_1000[1],acc_500_750[1],acc_250_500[1],acc_0_250[1],acc_background[1]]  
    plt.grid(True)
    plt.figure()
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2.-0.25, 1.01*height, '%.3f' % height)
    
    fig=plt.bar(range(len(num_list)), num_list,tick_label=name_list)  
    autolabel(fig)
    plt.ylabel('accuracy')
    plt.title(cell_line)
    file="D:/workspace of spyder/毕业设计/my project data/model_file/model_prediction_accuracy_plots_with_strength/"
    plt.savefig(file+cell_line+'_acc_with_strength.jpg')
    
    
predict_with_strength(model_DNd41, 'DNd41')
predict_with_strength(model_GM12878, 'GM12878')
predict_with_strength(model_H1hesc, 'H1hesc')
predict_with_strength(model_Helas3, 'Helas3')



