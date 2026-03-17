import numpy as np
from package.NN import sigmoid, softmax
from package.LF import cross_entropy_error_3
from package.gradient import numerical_gradient_2

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #初始化权重：符合标准正态分布的随机数数组（平均值为0，标准差为1）
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        #构建神经网络，用于预测
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        A1 = np.dot(x, W1) + b1
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, W2) + b2
        y = softmax(A2)
        return y

    def accuracy(self, x, t):
        #评估预测正确率：x：输入数据， t: 监督数据
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(t.shape[0])
        return accuracy

    def loss(self, x, t):
        #损失函数：x：输入数据， t: 监督数据
        y = self.predict(x)
        return cross_entropy_error_3(y, t) #批量处理，还未取得对应标签索引，需要先转换取得最大值对应的索引

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient_2(loss_W, self.params['W1']) #批量处理，数值微分法求梯度
        grads['b1'] = numerical_gradient_2(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_2(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_2(loss_W, self.params['b2'])
        return grads

#创建实例
net = TwoLayerNet(784, 100, 10)
# print(net.params['W1'].shape) #(784, 100)
# print(net.params['b1'].shape) #(100,)
# print(net.params['W2'].shape) #(100, 10)
# print(net.params['b2'].shape) #(10,)
#伪输入数据
x = np.random.rand(100, 784) #伪数据100个
y = net.predict(x)
# print(y.shape) #(100, 10)
#伪正确解标签
t = np.random.rand(100, 10) #伪数据100个
grads = net.numerical_gradient(x,t)
# print(grads['W1'].shape) #(784, 100)
# print(grads['b1'].shape) #(100,)
# print(grads['W2'].shape) #(100, 10)
# print(grads['b2'].shape) #(10,)
#评估正确率
accuracy = net.accuracy(x,t)
# print(accuracy) #0.12