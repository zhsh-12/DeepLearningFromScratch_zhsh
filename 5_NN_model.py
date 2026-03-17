import numpy as np
from package.NN import sigmoid, identity_function

#权重、偏置的初始化，保存在字典变量network中
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.06]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

#封装输入信号转换为输出信号的处理过程
def forward(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    A1 = np.dot(x,W1)+b1
    z1 = sigmoid(A1)
    A2 = np.dot(z1,W2)+b2
    z2 = sigmoid(A2)
    A3 = np.dot(z2,W3)+b3
    y = identity_function(A3)
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y) #[0.31529574 0.69283715]