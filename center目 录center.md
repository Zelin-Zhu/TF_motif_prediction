# <center>目 录</center>

# 摘 要

# ABSTRACT

# 第一章 文献综述

## 1 转录因子及其可识别序列的研究

### 1.1    转录因子的研究进展

​		转录因子（Transcription Factor）是一种与DNA序列结合控制基因转录的蛋白。目前关于转录因子的研究已经在揭示转录因子是如何在在基因表达中发挥作用，如何鉴别转录因子，转录因子识别的模体（motif),即转录因子特异性识别的一小段DNA序列，不同转录因子之间的相互作用，以及转录因子和疾病的关系等方面取得了一定的成果。

​		转录因子一般存在与真核细胞中，转录因子能够与RNA聚合酶结合形成转录起始复合体，进而控制转录的起始过程。有些转录因子是具有组织细胞特异性，在特异的组织细胞中，或者收到特异的分子影响后这些转录因子才会发生作用控制一些蛋白的特异表达，也就是说在一般不同的组织细胞中同一个转录因子发挥的作用是不同的。不过目前发现在人体细胞中少数的转录因子会在各种不同的细胞中结合大部分对应的模体。Fu, Y等人在2008年发现CTCF转录因子能够在其测试的所有细胞系中识别大部分在整个基因组上14000个左右长14bp的模体。

​		转录因子作为一种DNA结合蛋白，可以采用通用的DNA蛋白鉴别方式对其进行鉴别，常见的方法有单杂交试验（one-hybrid assays），DNA富集纯化质谱（ DNA affifinity purifification\-mass spectrometry）以及蛋白微阵列（protein microarrays ），目前大部分人们熟知的转录因子是通过判断识别序列与已知的DNA结合结构域(DBD DNA Binding Domain)的同源性来鉴别的。

​		不同的转录因子之间在结合DNA位点时会存在一些协同作用和一些影响机制。不管是从理论上来说还是从实验的角度这一观点都表明不同的转录因子共同对基因表达产生影响。有很多解释转录因子之间相互影响的方式，一种是受蛋白之间的相互作用，两种或两种以上转录因子在DNA上的结合会带来更高的结合稳定性，一种是DNA被某种转录因子结合之后会在结合位点附近被改变其空间结构进而影响其他转录因子的结合。高通量的胞外实验表明转录因子的协同结合会影响转录因子复合体与位点结合的<u>（偏好，强度，preference）</u>

​	    尽管人们对于转录因子的作用机制和转录因子本身抱有极大的研究兴趣，对于转录因子的研究也正在不断深入，但目前的研究还不能从实验的角度对于特定的转录因子在特定细胞系下准确给出其在整个基因组上的所有识别位点，以及转录因子最终是如果影响和调节哪些基因的转录和表达。

### 1.2 研究转录因子可识别序列的方法

#### 1.2.1 研究转录因子可识别序列的传统方法

​		在转录因子的研究中，关于转录因子识别的模体的研究是十分关键的，研究清楚转录因子在DNA上结合的模体能够帮助进一步有效地研究转录因子在基因表达中发挥的作用，不同转录因子之间以及转录因子与其他DNA结合蛋白之间的相互作用。传统的研究转录因子的motif的方式是通过计算一段序列权重位置矩阵（PWM Position wight Matrix)来计算转录因子对这段序列的结合能力。PWM方法在序列的每一个基因序列位置上给ATCG四种碱基分别定义了一个得分，将这些得分做乘法运算得到的积作为这段序列对转录因子的被识别概率值。

介绍PWM的优缺点

#### 1.2.2 卷积神经网络在研究转录因子可识别序列中的应用

 		在近几年深度学习的方法被引入到转录因子可识别序列预测当中，2015年Babak Alipanahi，Matthew T Weirauch等人，利用卷积神经网络建立了一系类的模型分别对一些识别DNA和RNA序列的蛋白对DNA和RNA序列进行预测。其中包括一些转录因子识别DNA序列的模型。这些模型能够将一段固定长度的碱基序列分类为一个特定的转录因子在一种细胞环境下能识别该序列或不能识别该序列。每个单独的卷积神经网络模型都是基于一个Chip-seq实验数据训练的，因此在不同的数据下模型的结构有所不同，另外他们在训练模型和验证模型时只采用了Chip-seq实验数据中，转录因子识别效果最明显的1000段数据，因此他们的模型虽然准确率较高，但模型忽略了占比为更多数的识别效果一般或者识别效果较弱的数据。

​		2017年Qian Qin, Jianxing Feng等人在利用深度学习进行转录因子可识别序列预测时建立了在多细胞下，多种转录因子的可识别序列预测问题，他们的工作解决了Babak Alipanahi，Matthew T Weirauch等人建立的模型只能在单个细胞环境下，对单个转录因子的可识别序列进行二分类的预测。他们通过Embedding的方式，将不同的细胞系和转录因子编码为分别编码为固定长度的向量，分别作为不同的部分输入到卷积神经网络中，最终给出一条序列在一个细胞环境下能够被一种转录因子所识别的结果。



​		介绍一下chip-seq原理及数据。

## 2 本研究的目的和意义

​	Babak Alipanahi，Matthew T Weirauch等人以及Qian Qin, Jianxing Feng的转录因子可识别序列预测研究做到了对一段序列进行二分类的工作，他们的模型能够判断一段固定长度的碱基序列能否被转录因子识别，在他们的模型中这些固定长度的序列长度一般在100bp左右，但研究表明大部分转录因子可识别的位点长度在10到20bp左右。也就是说他们的研究能够判断一段100bp左右长度的序列是否包含一段10到20bp左右长度的可识别位点motif。本研究的目的指在不仅要判断一段固定长度的序列是否包含motif，还要在包含motif的序列中，预测出motif在这段序列中的位置。这样能够帮助实验研究人员更好的研究转录因子对识别DNA序列。本研究只针对一种转录因子CTCF,对该转录因子在四种不同细胞系下的序列识别进行预测，针对每个细胞系构建相同结构的模型并在不同细胞系之间进行交叉预测，分析CTCF转录因子在不同细胞系下对序列识别的差异。同时本研究将不同识别效果的转录因子可识别序列都加入到卷积神经网络的训练样本中，提高模型对不同识别效果序列预测的通用性。另外本文还对预测出的motif进行突变预测，找出最容易引发转录因子不可识别的突变位点及其突变方式，对转录因子识别位点突变可能带来的危害研究提供支持。

### 2.2 本文做的工作和采取的方法

​		本文采用的数据为Chip-seq数据，染色质免疫共沉淀技术（Chromatin Immunoprecipitation，ChIP）也称结合位点分析法，这种技术通过在细胞内切割DNA,破碎后细胞后，选择DNA结合蛋白的免疫物质将结合蛋白识别的序列沉淀。进一步可通过测序找到对应的结合序列。本文采用的数据来自ENCODE项目数据库http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/

​		本文首先基于CTCF转录因子在四个细胞系下的Chip-seq实验数据分别建立CTCF可识别序列的二分类模型以达到预测一段固定长度的序列能否在相应的细胞系下被CTCF转录因子识别。在二分类模型建立完成后，通过对可识别序列进行局部保留，并替换序列的其他部分后输入二分类模型，找到使得经过替换后的序列依然能够通过二分类模型被预测为可识别序列的部分作为预测的motif。

​		 通过将motif的各个位点的碱基分别进行替换突变或丢失突变并将序列输入到二分类模型中，找出使得可识别序列突变为不可识别序列的突变位点和对应的突变方式，作为危险突变的预测。

​         通过在四个细胞系下建立的二分类模型交叉预测在其他三个细胞系的测试样本和训练样本分析交叉预测的准确率。找出CTCF转录因子在不同细胞系下识别DNA序列可能表现出的规律。

​       

# 第二章  卷积神经网络

## 1卷积神经网络结构

### 1.1卷积神经网络结构图

卷积神经网络一般由输入层，卷积层，池化层，全连接层以及输出层构成

![image-20210415111821529](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20210415111821529.png)

### 1.2 输入层

​        在我们的卷积神经网络的卷积神经网络要实现对一段核酸序列的的二分类，在我们将核酸序列信息输入神经网络时首先需要对序列进行编码，将包含碱基ATCG的字符序列转化为数值矩阵。在这里我们采用one-hot编码，对于每个碱基座位我们给定ATCG四个碱基类型通道，当该座位出现对应的碱基时将该碱基类型的通道赋值为1其他通道赋值为0。在一些特殊情况下，碱基座位上的碱基是不明确的，我们将不明确的碱基定义为N,当碱基座位上的碱基为N时，该座位上的所有碱基通道都被赋值为0.25。

<center>                                        <img src="C:\Users\ASUS\Desktop\图片1.png" alt="图片1" style="zoom: 33%;" /></center>

​		在将核酸序列输入卷积神经时我们不是每次只输入一条序列而是一次输入n条序列，并且我们需要固定输入序列的长度l。以此我们的输入层是由一个l $\times $ 4$\times$ n的矩阵构成的

<img src="C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20210415130840858.png" alt="image-20210415130840858" style="zoom:67%;" />

### 1.2卷积层

​		卷积神经网络的卷积层利用该层的卷积核通过对输入层的输入进行卷积运算后输出到下一层，一个卷积神经网络可包含多个卷积层，一个卷积层可包含多个卷积核。一个卷积核是一个二维的矩阵，矩阵的列数等于输入序列的列数，当输入序列为编码后的核酸序列时卷积核的列数Cl=4,矩阵的行数Cr是一个可变的参数。p$\times$ q的矩阵A和相同大小的矩阵B卷积运算得到一个数c。c=$\Sigma_{i=1}^{p}\Sigma_{j=1}^{q}a_{i,j}\times b_{i,j}$通过将一个卷积核在序列上滑动并不断进行矩阵卷积运算，得到一个l*1的矩阵。

![图片2](C:\Users\ASUS\Desktop\图片2.png)

当卷积核为k个时，对于个序列矩阵利用每个卷积核经过同样的运算后将结果拼接在一起得到一个l*k的矩阵。

<img src="C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20210415130325617.png" alt="image-20210415130325617" style="zoom:50%;" />

对于n条序列经过卷积层后输出为一个l$\times$k$\times$n的矩阵

<img src="C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20210415131212515.png" alt="image-20210415131212515" style="zoom:67%;" />

### 1.3 池化层.

​     池化层的输入是卷积层的输出，池化层一般采取两种池化方法，全局平均池化和全局最大池化。一个碱基序列构成的矩阵经过一个卷积核的运算后得到一个l$\times$ 1的向量，对于全局平均池化，输出是这个向量各个位置的平均值，对于全局最大池化输出是这个向量各个位置上的最大值。对于k个卷积核卷积运算后的向量拼接成的矩阵，全局平均池化和全局最大池化分别输出矩阵每一列的平均值和最大值。因此在n条序列经过卷积层运算后输入池化层后，池化层的输出为k*n的矩阵

![image-20210415140055333](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20210415140055333.png)

### 1.4 全连接层

​		一个全连接层由权重矩阵W和m个激活单元构成。当一个全连接层的输入是一个k*1的向量v时权重矩阵的大小为m$\times$k。经过W$\times$v运算得到另外一个m*$\times$1的向量u。u每个位置上的元素作为激活单元的输入经过激活单元内relu函数后作用后输出。ReLu函数定义为f(x)=0(x<=0),f(x)=x(x>0)。因此一个k*1的向量经过带有m个激活单元的全连接层后输出为一个m * 1的向量。可以将上一个全连接层的输出作为下一个全连接层的输入拼接在一起形成一个全连接网络。对于一条序列二分类卷积神经网络模型只输出一个值，因此全连接网络的最后一层只有一个激活单元。对于多分类模型全连接神经网络最后一层的机会单元个数与分类的类别数目相同。

### 1.5 输出层

​		全连接层的输出一般不直接作为最终的输出，而是经过输出层函数的作用最为最后的输出。一般采用sigmoid函数和softmax函数分别作为二分类和多分类的输出层函数。当进行二分类时，全连接层最后输出一个值x最作为输出层的输入,sigmoid函数定义为$f(x)=\frac{1}{1+e^{-x}}$,此时输出层的输出为y=f(x)。当进行多p分类时，全连接层最后输出一个p*1的向量Z，softmax函数定义为$Y_i=softmax(Z_i)=\frac{e^{Z_i}}{\Sigma_{j=1}^{p}e^{Z_j}}$。 此时输出层输出为p * 1的向量 Y

## 2 训练卷积神经网络

​		在建立卷积神经网络的过程中，有一些参数是人为设定的，称为超参数，如卷积核矩阵的行数，卷积层的个数和池化层的个数以及全连接层的隐藏单元数，全连接层的层数。这些参数决定了模型的结构。还有一些参数是需要训练的我们称这些参数为可训练参数，如卷积核矩阵，全连接层矩阵。卷积神经网络通过正向传播和反向传播来训练可训练参数的。通过将训练样本编码后输入到输入层然后一步步计算最后输出到输出层的过程叫正向传播。通过计算输出层和训练样本标签的误差计算从输出层反向一步步计算各层参数关于误差的梯度的过程叫反向传播。在反向传播计算完个参数关于误差的梯度后，根据设置的学习率，调整参数。

### 2.0 随机初始化

​		在正向传播之前，我们对于可训练参数要进行初始化，好的初始化方法能够帮助更快的训练模型，在训练神经网络的过程中可能会出现梯度消失或者梯度爆炸的问题而好的初始方法也能够尽可能的避免这两个问题。在这里我们采用Xavier均匀分布初始化【4】，该初始化方法使得可训练参数服从均值为0，标准差为$\sqrt{\frac{2}{n_j+n_{j+1}}} $的均匀分布。$W_j$~U(-$\sqrt{\frac{6}{n_j+n_{j+1}}},\sqrt{\frac{6}{n_j+n_{j+1}}}$)  其中n_j为第输入第j层的参数个数，n_j+1为第j+1层参数的输入个数即第j层参数的输出个数。

### 2.1损失函数及反向传播

​		神经网络的损失函数和反向传播是训练神经网络的关键要素，训练神经网络的过程就是为了让正向传播的输出结果和样本标签的损失函数变得更小，反向传播的过程就是在计算损失函数关于各个可训练参数的梯度。在二分类的模型中，模型的输出在输出层经过softmax函数变为一个[0,1]区间上的数y,训练样本的标签被标记为$\hat{y}$ 。损失函数描述了模型输出y与目标值$\hat{y}$的差距。我们采用交叉熵来表示损失函数。交叉熵（Cross Entropy）是Shannon在信息论提出的一个用于度量两个概率分布间的差异性信息的概念。我们定义损失函数为J 则J=$\hat{y}ln(y)$

​      反向传播则是通过求导的链式法则在计算损失函数J关于各个可训练参数的梯度，求导过程从输出层开始先求损失函数关于输出层参数的梯度$\frac{\part J}{\part W_L}$，进一步利用输出层参数的梯度和上一层参数关于输出层参数的函数求上一层参数关于输出层参数的梯度$\frac{\part W_L}{\part W_{L-1}}$，这该层参数关于损失函数的梯度$\frac{\part J}{\part W_{L-1}}=\frac{\part J}{\part W_L}\times \frac{\part W_L}{\part W_{L-1}}$,以此类推损失函数关于第i层的梯度$\frac{\part J}{\part W_1}=\frac{\part J}{\part W_L}\prod_{l=i}^{L-1}\frac{\part W_l}{\part W_{l-1}}$

### 2.2 dropout和正则化

​		 在训练神经网络的过程中，可能存在过拟合的现象，即模型对在训练集上的表现很好而在验证集上的表现很差。这种过拟合现象的产生可能是由多方面的因素造成的，其中一种原因是模型对于训练集中的局部特征十分敏感，缺乏对全局特征的把握，在模型中表现为模型在某些层的参数方差过大，使得模型的输出受某些节点的影响过大。解决过拟合的常见方法有在全连接层中加入dropout层，以及采用L1正则化和L2正则化。

​		dropout 层的节点个数与位于该dropout层的上一层的全连接层的激活单元个数相同。每一个Dropout层节点的输入等于上一层全连接层对应节点的输出并以一定的概率输出该节点输入或0

<img src="C:\Users\ASUS\Desktop\图片1.png" alt="图片1" style="zoom: 33%;" />

​	正则化操作是可训练参数的范数也加入到损失函数当中。我们设没有正则化操作时的损失函数值J0,加入正则化操作或的损失函数值为J，每一层的可训练参数为$W_l$, 共L层可训练参数，则：

L1正则化：$J=J_0+\Sigma_{l=1}^{L}\beta_l\Sigma_{w_i\in W_l} |w_i|$

L2正则化：$J=J_0+\Sigma_{l=1}^{L}\beta_l\Sigma_{w_i\in W_l} w_i^2$

### 2.3 批训练和梯度优化函数

​       在训练模型的过程中，通常不是在输入一个样本后计算损失函数值，然后进行反向传播计算梯度后就调整可训练参数，而是输入一批样本，这批样本通常是总样本一小部分，这批样本的数量称为batchsize。分别计算这批样本中每个样本正向传播的损失函数值和每个样本对应的可训练参数要调整的值$\Delta_i$，最后计算这一批样本带来的对可训练参数要调整的值平均值avg$\Delta_i$ 后根据平均值调整参数。这样做不仅助于加快模型的训练而且当训练样本数据很大可以解决无法一次将所有样本输入到计算机内存中的问题。另外在训练过程中我们不仅只对每个样本训练一次或几次而是训练及时上百次，甚至成千上万次。训练的次数和模型的规模也就是模型的可训练参数的个数有关，也与样本的构成有关，如果样本存在较多的冗余则需要训练的次数会少一些，反之需要更多的训练次数。我们把所有样本经过模型训练的次数称为epoch。

​		在根据梯度调整参数的过程中我们一般不直接根据每一批样本的梯度进行调整，而是通过梯度优化函数，在每一次参数的调整中加入之前的梯度。这里我们采用RMSprop梯度优化函数来计算我们每一次参数的调整。RMSprop是一种针对mini-batch训练的梯度优化函数，其针对每个batch的产生调整量$\Delta\theta$

定义全局学习率$\epsilon$ ，默认为0.001，衰减数率$\rho$ ,默认为0.9，可训练参数为$\theta$, 常数$\delta=10^{-7}$用于保持除法的数值稳定性，初始冲量r=0。

计算各参数关于损失函数在m个小批量样本的平均梯度g

计算累计平方梯度$r(t+1)=\rho r(t)+1-\rho g^2$

计算参数调整量$\Delta \theta=-\frac{\epsilon}{\sqrt{\delta+r}}\times g$

### 2.4 callback



# 第三章  转录因子可识别序列预测

## 3.1二分类模型

​      为了预测转录因子的识别位点（motif),我们首先利用tensorflow框架建立二分类模型。这个二分类模型能够判断一条固定长度的基因序列是否包含可被转录因子CTCF识别的位点,即是否为转录因子的可识别序列。为了考虑模型在不同细胞环境下的泛化能力，我们分别在四个细胞系下(DNd41,GM12878,H1hesc,Helas3) 建立关于CTCF可识别序列的二分类模型。

### 3.1.0 数据预处理

​		在构造训练集合测试集之前我们需要对Chip-seq 数据进行预处理，ENCODE数据库中提供的narrowpeak文件中给出了转录因子在整个基因组上的可识别序列，这些序列用染色体坐标来进行表示（染色体，起始位点，终止位点）其中还有一列peak表示从起始位点开始到测序过程中该区间的序列对应碱基作为重复最高的位置。我们选择以peak为中心的长度为100的序列作为正面样本序列。通过bedtools工具将坐标从人类基因组数据hg19上转化为对应的碱基序列。

​		在负面样本的构造中，与Babak Alipanahi，Matthew T Weirauch等人采用随机序列的方法不同，我们采用基因组上的不可识别序列作为负面样本，这样能够使负面样本不具备明显的随机统计特征，使得模型能更好的从基因组序列上分类可识别序列和不可识别序列。我们chip-seq数据中根据转录因子的对序列的识别强度分四个区间分别选取了2000条正面数据样本共8000条正面数据样本。另外选取10000条负面数据样本。

​		我们将50%的正面样本作为训练集和验证集，训练集合测试集的划分比例为3:1，50%的样本作为测试集，在每个集合选取同样数量的负面样本。在训练之前对训练集合验证集进行随机乱序处理。

### 3.1.1 模型预训练

​		在正式训练最终可用的模型之前，我们需要对模型进行预训练，预训练时我们只采用相对较少的数据进行训练，并且尝试不同的模型结构和训练次数以确定适当的模型结构和参数作为最终的模型结构和参数。我们在四个细胞系下采用2000个训练样本，正面样本和负面样本各1000个来进行预训练。

我们首先尝试建立一个规模较小的模型model1并且以较小的epochs=50来进行训练。

模型的参数 和训练参数如下

detctor_length=5
num_detector=16
num_hidden_unit=16
weight_decay = 0.01

epochs=50

模型在四个细胞系样本下的表现

<img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\H1hesc_model1_128_50_train_curve.jpg" alt="H1hesc_model1_128_50_train_curve" style="zoom: 67%;" /><img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\DNd41_model1_128_100_train_curve.jpg" alt="DNd41_model1_128_100_train_curve" style="zoom: 67%;" />

<img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\GM12878_model1_128_50_train_curve.jpg" alt="GM12878_model1_128_50_train_curve" style="zoom: 67%;" /><img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\GM12878_model1_128_100_train_curve.jpg" alt="GM12878_model1_128_100_train_curve" style="zoom: 67%;" />

<img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\H1hesc_model1_128_50_train_curve.jpg" alt="H1hesc_model1_128_50_train_curve" style="zoom: 67%;" /><img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\H1hesc_model1_128_100_train_curve.jpg" alt="H1hesc_model1_128_100_train_curve" style="zoom: 67%;" />

<img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\Helas3_model1_128_50_train_curve.jpg" alt="Helas3_model1_128_50_train_curve" style="zoom:67%;" /><img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\Helas3_model1_128_100_train_curve.jpg" alt="Helas3_model1_128_100_train_curve" style="zoom:67%;" />

分析在epochs=50时模型在所有细胞系样本中的acc都有上升趋势，在GM12878和Helas3细胞系数据样本中的val_acc有上升趋势说明epochs<=50并不能训练出最佳的模型，而在epoch=100模型在各细胞系中val_acc开始出现下降或波动趋势说明此时模型已经过拟合，因此在epoch<=100时能够训练出最佳模型

另外模型的acc和val_acc均不高这表明模型并不能很好的拟合数据，可以尝试适当提高模型的复杂度。

我们建立model2,model2的参数

<img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\DNd41_model2_128_100_train_curve.jpg" alt="DNd41_model2_128_100_train_curve" style="zoom:67%;" /><img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\GM12878_model2_128_100_train_curve.jpg" alt="GM12878_model2_128_100_train_curve" style="zoom:67%;" />

<img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\H1hesc_model2_128_100_train_curve.jpg" alt="H1hesc_model2_128_100_train_curve" style="zoom:67%;" /><img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\Helas3_model2_128_100_train_curve.jpg" alt="Helas3_model2_128_100_train_curve" style="zoom:67%;" />

detctor_length=5
num_detector=32
num_hidden_unit=32
weight_decay = 0.01

这时我们发现模型的准确度并没有明显的提高，说明进一步提高模型复杂度对预测的准确度没有明显的效果

此时我们考虑到detctor_length可能对模型的表现有比较大的影响，从文献中了解到CTCF的识别motif在10-12bp左右，因此我们在motif识别长度范围上适当增大设置detector_length=16

于是我们建立model3

detctor_length=16
num_detector=32
num_hidden_unit=32
weight_decay = 0.01

<img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\DNd41_model3_128_100_train_curve.jpg" alt="DNd41_model3_128_100_train_curve" style="zoom: 200%;" /><img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\GM12878_model3_128_100_train_curve.jpg" alt="GM12878_model3_128_100_train_curve" style="zoom:67%;" />

<img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\H1hesc_model3_128_100_train_curve.jpg" alt="H1hesc_model3_128_100_train_curve" style="zoom:67%;" /><img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\Helas3_model3_128_100_train_curve.jpg" alt="Helas3_model3_128_100_train_curve" style="zoom:67%;" />



这是我们发现模型的准确度提高了百分之10左右acc 已经达到95%以上，val_acc已经达到85以上。motif_detector的长度和模型准确度的关系给我们一个启示，是否可以通过尝试其他转录因子在不同motif_detetor长度下模型的准确度来预测这些转录因子的识别motif长度。

不过此时模型的val_acc 与acc仍有一定差距，我们尝试给模型增加卷积层和全连接层同时dropout层

这是我们建立model4

detctor_length=16
num_detector=32
num_hidden_unit=32
weight_decay = 0.01

两个卷积层，连个全连接层和两个dropout层。

<img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\DNd41_model4_128_100_train_curve.jpg" alt="DNd41_model4_128_100_train_curve" style="zoom:67%;" /><img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\GM12878_model4_128_100_train_curve.jpg" alt="GM12878_model4_128_100_train_curve" style="zoom:67%;" />

<img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\H1hesc_model4_128_100_train_curve.jpg" alt="H1hesc_model4_128_100_train_curve" style="zoom:67%;" /><img src="D:\workspace of spyder\毕业设计\my project data\model_file\pretrainning_models\Helas3_model4_128_100_train_curve.jpg" alt="Helas3_model4_128_100_train_curve" style="zoom:67%;" />

通过分析发现model4的准确度表现与model3的准确度表现并没有显著的区别，这表明进一步增加模型层数对模型的提高没有显著帮助，因此我们选择模型结构更为简单的model3的模型结构作为最终的模型结构。

### 1.1模型结构图

​		我们基于tensorflow2.0建立二分类模型的卷积神经网络，在尝试不同的超参数组合后，找到较优的模型。我们选择的初始模型的超参数满足表3.1-1，模型的结构如图3.1-2

| 超参数                     | 值   |
| -------------------------- | ---- |
| 卷积核个数                 | 32   |
| 卷积核长度                 | 16   |
| 全连接层个数               | 3    |
| 第一个全连接层单元数       | 64   |
| 第一个全连接层L2正则化系数 | 0.01 |
| 第一个dropout层参数        | 0.5  |
| 第二个全连接层单元数       | 32   |
| 第二个全连接层L2正则化系数 | 0.01 |
| 第二个dropout层参数        | 0.5  |
| 第三个全连接层单元数       | 1    |

​																		          模型结构图：



### 1.2 不同细胞系下的二分类模型

​		在通过预训练确定模型的结构之后，我们需要进一步扩大训练样本，并且在训练过程中保存表现最佳的模型，已经采用earlystopping减少不必要的训练时间。这是我们在四个细胞系下采用共8000个正面样本和8000个负面样本来训练最终的二分类模型。

在四个细胞系下的模型训练曲线如下，对应的模型见附件

<img src="D:\workspace of spyder\毕业设计\my project data\model_file\DNd41_16000samples_log\DNd41_16000samples_train_curve.jpg" alt="DNd41_16000samples_train_curve"  />val_acc=0.93

![GM12878_16000samples_train_curve](D:\workspace of spyder\毕业设计\my project data\model_file\GM12878_16000samples_log\GM12878_16000samples_train_curve.jpg)val_acc=0.91

![H1hesc_16000samples_train_curve](D:\workspace of spyder\毕业设计\my project data\model_file\H1hesc_16000samples_log\H1hesc_16000samples_train_curve.jpg)val_acc=0.93

![Helas3_16000samples_train_curve](D:\workspace of spyder\毕业设计\my project data\model_file\Helas3_16000samples_log\Helas3_16000samples_train_curve.jpg)val_acc=0.89

分析：在四个细胞系下模型的准确度都达到90左右。经过分析我们发现模型在不同强度的序列的预测准确度上有较大的差异

![DNd41_acc_with_strength](D:\workspace of spyder\毕业设计\my project data\model_file\model_prediction_accuracy_plots_with_strength\DNd41_acc_with_strength.jpg)

![GM12878_acc_with_strength](D:\workspace of spyder\毕业设计\my project data\model_file\model_prediction_accuracy_plots_with_strength\GM12878_acc_with_strength.jpg)



![H1hesc_acc_with_strength](D:\workspace of spyder\毕业设计\my project data\model_file\model_prediction_accuracy_plots_with_strength\H1hesc_acc_with_strength.jpg)

![Helas3_acc_with_strength](D:\workspace of spyder\毕业设计\my project data\model_file\model_prediction_accuracy_plots_with_strength\Helas3_acc_with_strength.jpg)

我们发现模型在高强度序列上的表现很好准确度能够达到98%左右，在低强度序列上的表现相对较差，在背景序列的预测表现上也相对较好。

可能的原因：

* 高强度序列的序列特征更显著

如果高强度序列特征明显的话在模型训练的过程中模型会更倾向于提取高强度序列的特征而更少的提取低强度序列的特征。

* 高强度序列的数据集噪声较小

这可能是因为高强度序列特征明显在识别过程中受环境，实验条件等因素实验过程中不容易被干扰因此数据的噪音较小。而低强度序列特征不明显且在实验数据获取的过程中容易产生噪声。

猜想：对低强度序列单独建立模型可能会提高模型对低强度序列判断的准确度



## 3.2 motif预测与训练样本扩充

​        我们基于二分类模型来对可识别序列中的motif进行预测，利用二分类模型预测motif的前提是二分类模型的可靠性，由于二分类模型在高强度序列的二分类准确度很高，所以我们只考虑对各个细胞系的2000条高强度序列进行motif预测。

### 2.1序列片段替换与motif定位

为了预测转录因子CTCF可识别序列中的motif位置，我们将正面样本进行片段替换，然后输入模型，保证替换后的序列经过模型预测后的输出值和原序列的输出值误差范围在10%以内。

具体的替换流程如下：

​	0. 通过模型计算可识别序列seq的预测值

   1.设置motif位置的在可识序列seq中的左端点和右端点p和q，初始为p=1,q=100 ,mid=int(p+q/2)

2. 替换 序列的[p,mid]区间上的碱基为A得到序列seq_right
3. 如果seq_right 通过模型的可识别验证即(predict(seq)-predict（seq_right))/predict(seq)<0.1则p=mid，继续第2步
4. 如果seq_right没有通过则替换序列的[mid,q]为A得到seq_left,将seq_left输入模型检测是否通过验证，若通过则q=mid,继续第2步
5. 如果seq_left也没有通过验证，则替换  [p,p+(q-p)/4]和[q-(q-p)/4,q]区间上的碱基为A得到seq_mid,检测seq_mid是否通过验证，若通过验证则继续第二步，否则第六步
6. 将p位置上的碱基替换为A得到seq_ 若通过验证则p+1，继续第六步，若不通过进行第七步
7. 将q位置上的碱基替换为A得到seq^, 若seq^ 通过则 q-1继续第7步，否则截止替换输出p,q

### 2.2 motif预测结果及其分析

### 2.3 样本扩增



## 3 序列强度预测多分类模型



### 3.1分强度训练二分类模型交叉预测检验.

### 3.2 分步多分类预测与单步多分类预测



## 4 可识别序列敏感突变预测

​             我们将可识别序列中的motif区间依次进行替换和删除操作，找到使得可识别序列在模型的预测值低于0.5，即由使得可识别序列变为不可识别序列的的突变及其突变方式。在删除操作中，删除相应的位点后需要在序列的左端或者右端补充一个序列A,如果删除位置靠近左端则在右端补充，否则在左端补充。

# 第四章 结论与展望



创新点：



# 参考文献.

[1] Samuel A. Lambert, Arttu Jolma, Laura F. Campitelli, Pratyush K. Das, Yimeng Yin, Mihai Albu, Xiaoting Chen, Jussi Taipale, Timothy R. Hughes, Matthew T. Weirauch [The Human Transcription Factors] (https://www.sciencedirect.com/science/article/pii/S0092867418312571)Cell, Volume 175, Issue 2, 4 October 2018, Pages 598-599

[2] Alipanahi, Babak & Delong, Andrew & Weirauch, Matthew & Frey, Brendan. (2015). Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. Nature biotechnology. 33. 10.1038/nbt.3300. 

[3] Qin Q, Feng J (2017) Imputation for transcription factor binding predictions based on deep learning. PLoS Comput Biol 13(2): e1005403. https://doi.org/10.1371/journal.pcbi.1005403

【4】**Xavier Glorot, Yoshua Bengio**. Understanding the difficulty of training deep feedforward neural networks.*Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics*, PMLR 9:249-256, 2010.

# 附 录

# 致 谢