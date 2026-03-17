import numpy as np
from collections import OrderedDict
from package.layer import Affine, Relu, SoftmaxWithLoss
from package.gradient import numerical_gradient_2

#2层神经网络：构建、推理、识别精度、损失函数、误差反向传播法求梯度【方法2】
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        #生成层【按顺序】：第1层加权层、激活层；第2层加权层；最后输出softmax-with-loss层【体现：自由组合且按顺序的思想】
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
    def predict(self, x):
        #各层的正向传播：推理【推理过程不需要softmax层,正向传播至Affine2层】
        for layer in self.layers.values(): #这里上个字段返回的值，会自动进入下个字段【因为不断更新x】
            x = layer.forward(x)
        return x
    def accuracy(self, x, t):
        # 评估识别精度：x：输入数据， t: 监督数据
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(t.shape[0])
        return accuracy
    def loss(self, x, t):
        #损失函数：x：输入数据【截至Affine2层】， t: 监督数据
        x = self.predict(x) #获得来自Affine2层的结果，准备进入最后softmax-with-loss层
        return self.lastLayer.forward(x,t) #返回损失函数：cross_entropy_error_3【根据输入数据情况调整】
    def numerical_gradient(self, x, t):
        #数值微分法求梯度
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient_2(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_2(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_2(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_2(loss_W, self.params['b2'])
        return grads
    def gradient(self, x, t):
        #误差反向传播法求梯度
        #forward:最后softmax-with-loss层损失函数、传递梯度
        self.loss(x, t) #中间过程包含self.y、self.t【用于backward】
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout) #梯度，准备反向传播到下一层：Affine2
        #前面Affine层、激活层：损失函数、传递梯度
        layers = list(self.layers.values())
        layers.reverse() #更改反向传播顺序：Affine2 -> Relu -> Affine1
        for layer in layers:
            dout = layer.backward(dout) #这里输入最后softmax-with-loss层反向传播来的梯度
        # 设定：取出backward过程储存在Affine的权重参数、偏置参数的梯度
        grads = {}
        grads['W1'],grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'],grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

# #创建实例验证
# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
#
# # 创建测试数据
# x = np.random.randn(100, 784)  # 100个样本
# t = np.random.randn(100, 10)   # one-hot 标签
#
# # 调用 gradient 方法
# grads = net.gradient(x, t)
#
# # 检查返回的梯度
# print(f"W1 gradient shape: {grads['W1'].shape}")  # 应为 (784, 100)
# print(f"b1 gradient shape: {grads['b1'].shape}")  # 应为 (100,)
# print(f"W2 gradient shape: {grads['W2'].shape}")  # 应为 (100, 10)
# print(f"b2 gradient shape: {grads['b2'].shape}")  # 应为 (10,)






