import sys
sys.path.append(".")
import torch
import matplotlib.pyplot as plt
from package.deep_convnet_pytorch import DeepConvNetPyTorch
from package.trainer_pytorch import Trainer
from dataset.mnist import load_mnist
import time

#第1步：设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}") #Using device: cuda:0

start_time = time.perf_counter()

#第2步：导入数据
x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=False, one_hot_label=False)

#第3步：设置实验
max_epochs = 20
batch_size = 100
network = DeepConvNetPyTorch()
trainer = Trainer(network, device, x_train, t_train, x_test, t_test, batch_size, max_epochs)
trainer.train()

end_time = time.perf_counter()
print(f"model_training_time: {(end_time - start_time):.4f} second") #model_training_time: 346.3105 second

#第4步：保存模型
trainer.save_model('deep_convent_pytorch.pth')
print("Model saved to deep_convnet_pytorch.pth")

#第5步：绘制训练曲线
plt.plot(range(1, max_epochs+1), trainer.train_acc_history, label='Train')
plt.plot(range(1, max_epochs+1), trainer.test_acc_history, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()







