import sys
sys.path.append('.')  # 将项目根目录加入路径，以便导入dataset模块
from dataset.mnist import load_mnist

# 第一次调用：会解析原始文件并生成 .pkl，稍慢
(x_train, t_train, x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=False)

# 后续调用：直接加载 .pkl 文件，非常快
# (x_train, t_train, x_test, t_test) = load_mnist()  # 快速加载

print(f'训练图像形状: {x_train.shape}')  # 应为 (60000, 784)
print(f'训练标签形状: {t_train.shape}')  # 应为 (60000,)
print(f'测试图像形状: {x_test.shape}')   # 应为 (10000, 784)
print(f'测试标签形状: {t_test.shape}')    # 应为 (10000,)

#查看其中的数据
img = x_train[0]
label = t_train[0]
print(label) #5
print(img.shape) #(784,)
img = img.reshape(28, 28) #图像的形状变成原来的尺寸
print(img.shape) #(28, 28)
#显示图像：利用PIL工具
from PIL import Image
import numpy as np
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
img_show(img)