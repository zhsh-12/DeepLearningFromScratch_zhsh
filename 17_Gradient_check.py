import numpy as np
import sys
sys.path.append(".")
from dataset.mnist import load_mnist
from package.two_layer_net_2 import TwoLayerNet

#第1步：读取数据
x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=True, one_hot_label=True)

#第2步：构建网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
x_batch = x_train[:3]
t_batch = t_train[:3]

#第3步：数值微分法、误差反向传播法求梯度
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

#第4步：求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
# W1:5.045455742621769e-10
# b1:3.0474339703871323e-09
# W2:6.783500581122783e-09
# b2:1.3908233111176172e-07