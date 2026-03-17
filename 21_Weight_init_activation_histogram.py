import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def Relu(x):
    return np.maximum(x, 0)
def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000, 100) #1000个数据
node_num = 100 #各隐藏层的节点(神经元)数
hidden_layer_size = 5 #隐藏层有5层
activations = {}

X = input_data

for i in range(hidden_layer_size): #0,1,2,3,4
    if i != 0:
        X = activations[i-1] #将上一层激活值传入下一层作为输入值

    #改变初始值进行实验：
    # W = np.random.randn(node_num, node_num) * 1 #标准差为1的高斯分布
    # W = np.random.randn(node_num, node_num) * 0.01 #标准差为0.01的高斯分布
    # W = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) #Xavier初始值，适用sigmoid、tanh函数
    W = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num) #He初始值，适用Relu函数
    
    #改变激活函数的种类
    A = np.dot(X, W)
    Z = Relu(A)
    
    activations[i] = Z

#绘制直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1) #1行生成多个图
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([],[])
    plt.hist(a.flatten(), 30, range=(0, 1)) #划分为30个区间
plt.show()