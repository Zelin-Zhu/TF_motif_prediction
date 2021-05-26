# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:26:19 2021

@author: ASUS
"""
import vectorize_sequences as vs


file_path1="D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_seq_background.csv"
file_path2="D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_seq_250_500.csv"
file_path3="D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_seq_500_750.csv"
file_path4="D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_seq_750_1000.csv"
file_path5="D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_seq_0_250.csv"


VS_DNd41_background=vs.Vector_sequence(file_path1)
VS_DNd41_250_500=vs.Vector_sequence(file_path2)
VS_DNd41_500_750=vs.Vector_sequence(file_path3)
VS_DNd41_750_1000=vs.Vector_sequence(file_path4)
VS_DNd41_0_250=vs.Vector_sequence(file_path5)


digital_seq=VS_DNd41_background.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_background.npy')
digital_seq=VS_DNd41_250_500.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_250_500.npy')
digital_seq=VS_DNd41_500_750.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_500_750.npy')
digital_seq=VS_DNd41_750_1000.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_750_1000.npy')
digital_seq=VS_DNd41_0_250.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/DND41/DNd41_0_250.npy')
################################################################################################


file_path1="D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/GM12878_seq_background.csv"
file_path2="D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/GM12878_seq_250_500.csv"
file_path3="D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/GM12878_seq_500_750.csv"
file_path4="D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/GM12878_seq_750_1000.csv"
file_path5="D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/GM12878_seq_0_250.csv"


VS_GM12878_background=vs.Vector_sequence(file_path1)
VS_GM12878_250_500=vs.Vector_sequence(file_path2)
VS_GM12878_500_750=vs.Vector_sequence(file_path3)
VS_GM12878_750_1000=vs.Vector_sequence(file_path4)
VS_GM12878_0_250=vs.Vector_sequence(file_path5)


digital_seq=VS_GM12878_background.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/GM12878_background.npy')
digital_seq=VS_GM12878_250_500.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/GM12878_250_500.npy')
digital_seq=VS_GM12878_500_750.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/GM12878_500_750.npy')
digital_seq=VS_GM12878_750_1000.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/GM12878_750_1000.npy')
digital_seq=VS_GM12878_0_250.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/GM12878/GM12878_0_250.npy')
    
######################################################







file_path1="D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/H1hesc_seq_background.csv"
file_path2="D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/H1hesc_seq_250_500.csv"
file_path3="D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/H1hesc_seq_500_750.csv"
file_path4="D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/H1hesc_seq_750_1000.csv"
file_path5="D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/H1hesc_seq_0_250.csv"



VS_H1hesc_background=vs.Vector_sequence(file_path1)
VS_H1hesc_250_500=vs.Vector_sequence(file_path2)
VS_H1hesc_500_750=vs.Vector_sequence(file_path3)
VS_H1hesc_750_1000=vs.Vector_sequence(file_path4)
VS_H1hesc_0_250=vs.Vector_sequence(file_path5)


digital_seq=VS_H1hesc_background.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/H1hesc_background.npy')
digital_seq=VS_H1hesc_250_500.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/H1hesc_250_500.npy')
digital_seq=VS_H1hesc_500_750.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/H1hesc_500_750.npy')
digital_seq=VS_H1hesc_750_1000.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/H1hesc_750_1000.npy')
digital_seq=VS_H1hesc_0_250.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/H1hesc/H1hesc_0_250.npy')

#####################################3Helas3



file_path1="D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/Helas3_seq_background.csv"
file_path2="D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/Helas3_seq_250_500.csv"
file_path3="D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/Helas3_seq_500_750.csv"
file_path4="D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/Helas3_seq_750_1000.csv"
file_path5="D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/Helas3_seq_0_250.csv"




VS_Helas3_background=vs.Vector_sequence(file_path1)
VS_Helas3_250_500=vs.Vector_sequence(file_path2)
VS_Helas3_500_750=vs.Vector_sequence(file_path3)
VS_Helas3_750_1000=vs.Vector_sequence(file_path4)
VS_Helas3_0_250=vs.Vector_sequence(file_path5)


digital_seq=VS_Helas3_background.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/Helas3_background.npy')
digital_seq=VS_Helas3_250_500.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/Helas3_250_500.npy')
digital_seq=VS_Helas3_500_750.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/Helas3_500_750.npy')
digital_seq=VS_Helas3_750_1000.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/Helas3_750_1000.npy')
digital_seq=VS_Helas3_0_250.get_digital_sequences\
    ('D:/workspace of spyder/毕业设计/my project data/datafile/Helas3/Helas3_0_250.npy')

