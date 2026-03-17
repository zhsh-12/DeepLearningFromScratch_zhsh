import sys
sys.path.append(".")
import numpy as np
import matplotlib.pyplot as plt
from package.simple_convnet import SimpleConvNet
from package.trainer import Trainer
from dataset.mnist import load_mnist

#第1步：读取数据
x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=False, one_hot_label=False) #是否应该one-hot？
# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]

#第2步：设置实验
max_epochs = 20
network = SimpleConvNet(input_dim=(1,28,28), conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=max_epochs, mini_batch_size=100, 
                  optimizer='Adam', optimizer_param={'lr': 0.001}, evaluate_sample_num_per_epoch=1000)
trainer.train()

#第3步：保存参数
network.save_params("params.pkl")
print("Saved Network Parameters!")

#第4步：绘制图形
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="upper left")
plt.show()