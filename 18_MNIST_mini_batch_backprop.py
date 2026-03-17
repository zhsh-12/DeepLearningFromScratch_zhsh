import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from dataset.mnist import load_mnist
from package.two_layer_net_2 import TwoLayerNet

#第1步：导入数据集
x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=True, one_hot_label=True) #已调整为one-hot标签，调整以下损失函数种类
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#第2步：设置超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

#平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

train_loss_list =[]
train_acc_list =[]
test_acc_list = []

#第3步：随机选取训练数据、计算梯度、更新参数【随机梯度下降法】
for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度：误差反向传播法
    grad = network.gradient(x_batch, t_batch)

    #更新参数
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]

    #记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc |" + str(train_acc) + "," + str(test_acc))

#第4步：绘制train_loss与iters_num的关系图
plt.figure(figsize=(10,5))
x = np.arange(iters_num)
plt.plot(x, train_loss_list)
plt.xlabel('iteration')
plt.ylabel('train_loss')
plt.ylim(0,20)
plt.show()