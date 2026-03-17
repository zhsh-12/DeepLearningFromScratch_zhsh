import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from dataset.mnist import load_mnist
from package.multi_layer_net_extend import MultiLayerNetExtend
from package.optimizer import SGD

#第1步：读入MNIST数据
x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=True, one_hot_label=True)
x_train = x_train[:1000]
t_train = t_train[:1000]

#第2步：设置实验参数
max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

#第3步：构建、训练、评估
def __train(weight_init_std):
    #构建2个神经网络，对应:存在Batch Norm，不存在Batch Norm
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100,100,100,100,100], output_size=10,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100,100,100,100,100], output_size=10,
                                     weight_init_std=weight_init_std, use_batchnorm=False)
    #相同参数更新方法：优化器SGD
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(train_size/batch_size, 1)
    epoch_cnt = 0

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
            
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
    return train_acc_list, bn_train_acc_list

#第4步：绘制图形
weight_scale_list = np.logspace(0, -4, num=16) #生成16个对数间隔的数值，范围从10的0次方到10的-4次方
x = np.arange(max_epochs)
for i, w in enumerate(weight_scale_list):
    print("=====" + str(i+1) + "/16" + "=====")
    train_acc_list, bn_train_acc_list = __train(w)

    plt.subplot(4,4,i+1) #生成4行4列的子图网络，当前实验结果绘制在第i+1个子图中
    plt.title("W: "+ str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2) #只在最后一个子图显示图例标签
        plt.plot(x, train_acc_list, linestyle='--', label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle='--', markevery=2)
    plt.ylim(0, 1.0) 
    if i % 4 : #只在第1列显示y轴标签设置
        plt.yticks([]) 
    else:
        plt.ylabel('accuracy')
    if i < 12: #只在最后1行显示x轴标签设置
        plt.xticks([])
    else:
        plt.xlabel('epochs')
    plt.legend(loc='lower right')
plt.show()
   

    