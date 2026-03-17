import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 或 ['SimHei'] #黑体
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import sys, os
sys.path.append(".")

#第1步：定义数据增强流程：随机仿射变换，一次性组合旋转、平移、缩放和错切 【注意：MNIST是单通道灰度图像】
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=15, #随机旋转角度范围：+-15° 【模拟不同人的书写笔记倾斜】
        translate=(0.1, 0.1), #随机水平和垂直平移最多10% 【模拟数字在图像框内的位置变化】
        scale=(0.9, 1.1), #随机缩放范围：90%到110% 【模拟数字书写的大小差异】
        shear=10 #随机错切（剪切）角度：+-10°
    ),
    transforms.ToTensor(), #将PIL图像或numpy数组转换为Pytorch张量，并将像素值从[0,255]归一化到[0,1]
    transforms.Normalize((0.1307,),(0.3081,)) #使用MNIST的均值和标准差进行标准化,加速模型收敛，使数据分布更稳定
])
#仅进行基础转换，用于对比
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

#第2步：设置数据访问路径
data_root = "./data"
# raw_dir = os.path.join(data_root, 'MNIST', 'raw')
# print(raw_dir) #./data\MNIST\raw
# if os.path.exists(raw_dir):
#     print("找到数据")
#     print(os.listdir(raw_dir))
# else:
#     print("未找到数据")

#第3步：加载本地数据集
# 重要：root 指向 './data'，torchvision 会自动查找 ./data/MNIST/
train_dataset = MNIST(root=data_root, train=True, download=False, transform=train_transform) # 指向data目录，而非MNIST目录，download必须设为 False
test_dataset = MNIST(root=data_root, train=False, download=False, transform=test_transform)
# print("数据集加载成功")
# print(f"训练集样本数：{len(train_dataset)}") #60000
# print(f"测试集样本数：{len(test_dataset)}") #10000

#第4步：可视化增强效果
def visualize_augmentation(dataset, num_examples=8):
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for idx, ax in zip(indices, axes.flat):
        img_tensor, label = dataset[idx]
        #反标准化以便显示
        mean = torch.tensor([0.1307])
        std = torch.tensor([0.3081])
        img_display = img_tensor * std.view(1, 1, 1) + mean.view(1, 1, 1)
        img_display = img_display.squeeze().numpy()
        ax.imshow(img_display, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'标签：{label}', fontsize=12)
        ax.axis('off')
    plt.suptitle('MNIST数据增强示例(每个epoch看到的图像都不同)', fontsize=14)
    plt.tight_layout()
    plt.show()

#第5步：显示训练数据集的增强效果
# visualize_augmentation(train_dataset)
#对比：显示测试数据集的未作增强效果
# visualize_augmentation(test_dataset)

#第6步：创建数据加载器【已增强的训练集、未增强的测试集】
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) #不使用多进程加速
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True) 
# print("数据加载器创建完成")
# print(f"训练批次大小：{batch_size}") #64
# print(f"每个epoch的训练批次数：{len(train_dataset)}") #60000

#第7步：获得一个批次的已增强训练集数据
for images, labels in train_loader:
    print(f"图像张量形状：{images.shape} ") #[批次，通道，高，宽] 图像张量形状：torch.Size([64, 1, 28, 28])
    print(f"像素值范围：[{images.min():.3f},{images.max():.3f}]") #(标准化后) 像素值范围：[-0.424,2.821]
    print(f"标签张量形状：{labels.shape}") #[批次] 标签张量形状：torch.Size([64])
    break
# print("可以将train_loader和test_loader用于集成CNN模型训练")