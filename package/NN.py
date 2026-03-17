import numpy as np
#阶跃函数
def step_function_1(x):
    y = x > 0 #数组的每个元素均进行不等号运算，生成一个布尔型数组
    return y.astype(int) #将布尔型数组转换为int类型
def step_function(x):
    return np.array(x > 0, dtype=int)

#sigmoid函数：二元分类问题输出层
def sigmoid(x):
    return 1/(1+np.exp(-x)) #numpy的广播功能

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

#ReLU函数
def relu(x):
    return np.maximum(0, x) #选择最大值输出

def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad

#恒等函数：回归问题输出层
def identity_function(x):
    return x

#softmax函数：多元分类问题输出层
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0) #溢出对策：按行处理，因此提前转置矩阵
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))