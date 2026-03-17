import numpy as np
import sys
sys.path.append(".")
from dataset.mnist import load_mnist

#加载数据集
(x_train, t_train, x_test, t_test) = load_mnist(normalize=True, one_hot_label=True, flatten=True)
print(x_train.shape) #(60000, 784)
print(t_train.shape) #(60000, 10)

#使用mini-batch提取数据：特征和标签数据
train_size = x_train.shape[0] #获取训练数据行索引
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) #随机选取10个训练数据的行索引
x_batch = x_train[batch_mask] #获取对应的训练数据
t_batch = t_train[batch_mask] #获取对应的标签【前提：训练数据索引和标签索引一致】
print(x_batch.shape) #(10, 784)
print(t_batch.shape) #(10, 10)


