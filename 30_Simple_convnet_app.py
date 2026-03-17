import sys
sys.path.append('.')
from package.simple_convnet import SimpleConvNet
import numpy as np 

network = SimpleConvNet(input_dim=(1,10,10), conv_param={'filter_num': 10, 'filter_size': 3, 'pad': 0, 'stride': 1}, 
                        hidden_size=10, output_size=10, weight_init_std=0.01)
X = np.random.rand(100).reshape((1,1,10,10)) #创建100个随机浮点数一维数组，范围在[0,1)，形状为(100,)，重塑为4维数组(批次大小，通道数，高度，宽度)
T = np.array([1]).reshape((1,1)) #创建1个包含单个元素1的一维数组，形状为(1,)，重塑为2维数组

grad_num = network.numerical_gradient(X, T)
grad = network.gradient(X, T)

for key, val in grad_num.items():
    print(key, np.abs(grad_num[key] - grad[key]).mean())







    