import sys
sys.path.append(".")
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from package.deep_convnet_2 import DeepConvNet
from package.trainer import Trainer

#第1步：读入数据
x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=False, one_hot_label=False) #是否应该one-hot？
# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]

#第2步：设置实验
max_epochs = 20
network = DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=max_epochs, mini_batch_size=100, 
                  optimizer='Adam', optimizer_param={'lr': 0.001}, evaluate_sample_num_per_epoch=1000)
trainer.train() #59.35%

#第3步：保留参数
network.save_params("deep_convnet_2_params.pkl")   
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

import torch

print("=== GPU/CPU 诊断信息 ===")
# 1. 检查CUDA（即NVIDIA GPU支持）是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
# 2. 检查可用的GPU数量
print(f"可用的GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    # 3. 获取当前使用的GPU索引和名称
    current_device = torch.cuda.current_device()
    print(f"当前使用的GPU索引: {current_device}")
    print(f"当前GPU名称: {torch.cuda.get_device_name(current_device)}")
    # 4. (可选) 查看显存使用情况
    total_mem = torch.cuda.get_device_properties(current_device).total_memory / 1e9
    used_mem = torch.cuda.memory_allocated(current_device) / 1e9
    print(f"GPU显存总量: {total_mem:.2f} GB")
    print(f"已用显存: {used_mem:.2f} GB")
else:
    print("CUDA不可用。当前正在使用CPU进行训练。")

# 5. (关键) 检查模型参数所在的设备
# 假设你的网络实例叫 `network`，请替换为你实际的变量名
try:
    # 获取模型第一个参数所在的设备
    sample_param = next(network.parameters())
    print(f"\n模型参数所在设备: {sample_param.device}")
except AttributeError:
    print("\n无法直接获取网络参数，可能你的自定义框架结构不同。")
