import sys
sys.path.append(".")
import numpy as np 
import matplotlib.pyplot as plt
from package.simple_convnet import SimpleConvNet

def filter_show(filters, nx=8, margin=3, scale=10): #滤波器权重，每行显示滤波器的数量，子图之间的边距，缩放因子
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape #滤波器的4个维度
    ny = int(np.ceil(FN / nx)) #需要多少行显示所有滤波器，向上取整
    
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)#调整子图间距
    
    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[]) #创建ny*nx网格，从1开始计数，隐藏坐标轴刻度
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest') #显示第i个滤波器的第1个通道，使用反向灰度颜色映射，使用最近邻插值，保持像素边界清晰
    plt.show()

network = SimpleConvNet() 
#随机进行初始化后的权重
filter_show(network.params['W1']) 

#学习后的权重
network.load_params("params.pkl")
filter_show(network.params['W1'])
 
    


