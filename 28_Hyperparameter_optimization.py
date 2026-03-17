import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from dataset.mnist import load_mnist
from package.multi_layer_net import MultiLayerNet
from package.util import shuffle_dataset
from package.trainer import Trainer

#第1步：读入数据、重洗训练数据、分割验证数据
x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=True, one_hot_label=True)
x_train = x_train[:500]
t_train = t_train[:500]

validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

#第2步：构建网络、训练
def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100,100], output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val, epochs=epocs, mini_batch_size=100, optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()
    
    return trainer.train_acc_list, trainer.test_acc_list

#第3步：超参数的随机搜索【验证对象：学习率、权重衰减系数】
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    #指定搜索的超参数的范围
    lr = 10 ** np.random.uniform(-3, -2)
    weight_decay = 10 ** np.random.uniform(-8, -7)
    
    train_acc_list, val_acc_list = __train(lr, weight_decay)
    print("val_acc: " + str(val_acc_list[-1]) + " | lr: " + str(lr) + ", weight decay: " + str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" +str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list
    
#第4步：绘制图形
print("========== Hyper-Parameter Optimization Result ==========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0
for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val_acc:" + str(val_acc_list[-1]) + ") | " + key)
    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], '--')
    i += 1
    if i >= graph_draw_num:
        break
plt.show()
          
    

