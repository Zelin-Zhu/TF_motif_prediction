import numpy as np
import matplotlib.pyplot as plt

class Pre_trainning:
    def __init__(self,cell_line):
        self.cell_line=cell_line
        if cell_line=="DNd41":
            self.parent_file="DND41"
        else:
            self.parent_file=cell_line
        
        
        seqs_750_1000=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.parent_file+"/"+cell_line+"_750_1000.npy")
        seqs_500_750=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.parent_file+"/"+cell_line+"_500_750.npy")
        seqs_250_500=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.parent_file+"/"+cell_line+"_250_500.npy")
        seqs_0_250=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.parent_file+"/"+cell_line+"_0_250.npy")
        seqs_background=np.load("D:/workspace of spyder/毕业设计/my project data/datafile/"+self.parent_file+"/"+cell_line+"_background.npy")
        seqs_background=seqs_background[0:1000]
        seqs_positive=seqs_0_250[0:250]
        seqs_positive=np.append(seqs_positive,seqs_250_500[0:250],axis=0)
        seqs_positive=np.append(seqs_positive,seqs_500_750[0:250],axis=0)
        seqs_positive=np.append(seqs_positive,seqs_750_1000[0:250],axis=0)

        x_positive=seqs_positive
        y_positive=np.ones(len(x_positive))
        x_negtive=seqs_background
        y_negtive=np.zeros(len(x_negtive))
        y=np.append(y_positive,y_negtive,axis=0)
        x=np.append(x_positive,x_negtive,axis=0)

        index=[]
        for i in range(len(x)):
            index.append(i)
        
        np.random.shuffle(index)
        x=x[index]
        y=y[index]
        num=1500
        self.train_x=x[0:num]
        self.train_y=y[0:num]
        self.validate_x=x[num:-1]
        self.validate_y=y[num:-1]

    def pretrain(self,model,model_name,epochs,batch_size):
        self.history = model.fit(self.train_x,self.train_y,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.validate_x,self.validate_y)
                            )
                           
        plt.figure()
        plt.plot(self.history.epoch,self.history.history.get('acc'),label='acc')
        plt.plot(self.history.epoch,self.history.history.get('val_acc'),label='val_acc')
        plt.xlabel('epoches')
        plt.ylabel('accuracy')
        plt.title(self.cell_line+"_"+model_name)
        plt.legend()
        file="D:/workspace of spyder/毕业设计/my project data/model_file/pretrainning_models/"
        plt.savefig(file+self.cell_line+"_"+model_name+"_"+str(batch_size)+"_"+str(epochs)+"_train_curve.jpg")
       



