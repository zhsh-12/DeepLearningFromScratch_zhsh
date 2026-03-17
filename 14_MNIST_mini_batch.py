import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
sys.path.append(".") #为导入当前目录的文件而进行的设定
from dataset.mnist import load_mnist
from package.two_layer_net_1 import TwoLayerNet

#第1步：导入数据集
x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=True, one_hot_label=True)

#第2步：设置超参数
train_loss_list = []
iters_num = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#第3步：随机选取训练数据、计算梯度、更新参数【随机梯度下降法】
for i in range(iters_num):
    #获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #计算梯度：数值微分法
    # grads = network.numerical_gradient(x_batch, t_batch)
    grads = network.gradient(x_batch, t_batch)

    #更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grads[key]

    #记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

plt.figure(figsize=(10,5))
x = np.arange(iters_num)
plt.plot(x, train_loss_list)
plt.xlabel("iteration")
plt.ylabel("train_loss")
plt.ylim(0, 10)
plt.show()
