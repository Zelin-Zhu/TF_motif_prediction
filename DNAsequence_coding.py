import numpy as np
a
def seq_generator(length,num):
    seqs=np.zeros((num,length))
    np.random.seed(1234) #设置随机种子为1234
    seqs[i,:]=np.