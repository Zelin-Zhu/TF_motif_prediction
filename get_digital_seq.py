# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:26:19 2021

@author: ASUS
"""
import vectorize_sequences as vs

file_path1='D:/workspace of spyder/毕业设计/my project/datafile/DNd41_background_seq.csv'
file_path2="D:/workspace of spyder/毕业设计/my project/datafile/DNd41_250_500_seq.csv"
file_path3="D:/workspace of spyder/毕业设计/my project/datafile/DNd41_500_750_seq.csv"
file_path4="D:/workspace of spyder/毕业设计/my project/datafile/DNd41_750_1000_seq.csv"
file_path5="D:/workspace of spyder/毕业设计/my project/datafile/DNd41_0_250_seq.csv"


VS_DNd41_background=vs.Vector_sequence(file_path1)
VS_DNd41_250_500=vs.Vector_sequence(file_path2)
VS_DNd41_500_750=vs.Vector_sequence(file_path3)
VS_DNd41_750_1000=vs.Vector_sequence(file_path4)
VS_DNd41_0_250=vs.Vector_sequence(file_path5)


digital_seq=VS_DNd41_background.get_digital_sequences('DNd41_background.npy')
digital_seq=VS_DNd41_250_500.get_digital_sequences('DNd41_250_500.npy')
digital_seq=VS_DNd41_500_750.get_digital_sequences('DNd41_500_750.npy')
digital_seq=VS_DNd41_750_1000.get_digital_sequences('DNd41_750_1000.npy')
digital_seq=VS_DNd41_0_250.get_digital_sequences('DNd41_0_250.npy')
