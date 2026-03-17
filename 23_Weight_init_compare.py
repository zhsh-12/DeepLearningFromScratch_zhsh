import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from dataset.mnist import load_mnist
from package.optimizer import SGD
from package.multi_layer_net import MultiLayerNet
from package.util import smooth_curve

#第1步：读入MNIST数据
x_train, t_train, x_test, t_test = load_mnist(normalize=True,flatten=True, one_hot_label=True)
train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

#第2步：实验设置【3种初始化权重比较、相同参数更新优化器、每种权重对应一个神经网络，每个神经网络有相同层数和神经元数】】
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100], output_size=10, weight_init_std=weight_type) #激活函数：Relu
    train_loss[key] = []

#第3步：开始训练
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads) #获得更新的参数
        
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print("=====" + "iterations:" + str(i) + "=====")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))

#第4步：绘制图形
markers = {'std=0.01': "o", "Xavier": "s", "He": "D"}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0,2.5)
plt.legend()
plt.show()
    
