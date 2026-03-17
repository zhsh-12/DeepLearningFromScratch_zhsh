#3层神经网络的前向处理
import numpy as np
from package.NN import sigmoid, identity_function
#输入层 ——> 第1层
X = np.array([1.0, 0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])
print(W1.shape) #(2, 3)
print(B1.shape) #(3,)
print(X.shape) #(2,)
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1) #第1层激活函数
print(A1) #[0.3 0.7 1.1]
print(Z1) #[0.57444252 0.66818777 0.75026011]

#第1层 ——> 第2层
W2 = np.array([[0.1,0.4],[0.2, 0.5],[0.3, 0.6]])
B2 = np.array([0.1,0.2])
print(W2.shape) #(3, 2)
print(B2.shape) #(2,)
print(Z1.shape) #(3,)
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2) #第2层激活函数
print(A2) #[0.51615984 1.21402696]
print(Z2) #[0.62624937 0.7710107 ]

#第2层 ——> 第3层
W3 = np.array([[0.1,0.3],[0.2, 0.4]])
B3 = np.array([0.1,0.2])
A3 = np.dot(Z2, W3) + B3
Z3 = identity_function(A3)
print(A3) #[0.31682708 0.69627909]
print(Z3) #[0.31682708 0.69627909]