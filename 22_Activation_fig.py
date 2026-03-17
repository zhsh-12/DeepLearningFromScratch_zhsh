import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 或 ['SimHei'] #黑体
plt.rcParams['axes.unicode_minus'] = False

# 生成x值
x = np.linspace(-5, 5, 1000)

# 定义三个激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# 计算函数值
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

# 创建图形
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# 绘制Sigmoid函数
ax1.plot(x, y_sigmoid, 'b-', linewidth=2)
ax1.set_title('Sigmoid 函数', fontsize=14)
ax1.set_xlabel('x')
ax1.set_ylabel('σ(x)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-5, 5])
ax1.set_ylim([-0.1, 1.1])
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax1.axhline(y=1, color='k', linestyle='-', alpha=0.2)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.2)

# 绘制Tanh函数
ax2.plot(x, y_tanh, 'r-', linewidth=2)
ax2.set_title('Tanh 函数', fontsize=14)
ax2.set_xlabel('x')
ax2.set_ylabel('tanh(x)')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-5, 5])
ax2.set_ylim([-1.1, 1.1])
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax2.axhline(y=1, color='k', linestyle='-', alpha=0.2)
ax2.axhline(y=-1, color='k', linestyle='-', alpha=0.2)
ax2.axvline(x=0, color='k', linestyle='-', alpha=0.2)

# 绘制ReLU函数
ax3.plot(x, y_relu, 'g-', linewidth=2)
ax3.set_title('ReLU 函数', fontsize=14)
ax3.set_xlabel('x')
ax3.set_ylabel('ReLU(x)')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([-5, 5])
ax3.set_ylim([-0.1, 5.1])
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax3.axvline(x=0, color='k', linestyle='-', alpha=0.2)

plt.tight_layout()
plt.show()