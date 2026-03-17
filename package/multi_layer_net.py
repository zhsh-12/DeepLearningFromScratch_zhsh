import numpy as np
from collections import OrderedDict
from package.layer import Sigmoid, Relu, Affine, SoftmaxWithLoss
from package.gradient import numerical_gradient_2

#多层神经网络：构建、推理、识别精度、损失函数、数值微分法/误差反向传播法求梯度
class MultiLayerNet:
    """全连接的多层神经网络：输入层大小、隐藏层列表、输出层大小
    激活函数、指定权重的标准差等"""
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu', weight_init_std='relu',weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}
        #初始化权重
        self.__init_weight(weight_init_std)
        #构造层：加权层+激活层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)], self.params['b'+str(idx)])
            self.layers['Activation_function'+str(idx)] = activation_layer[activation]()
        idx = self.hidden_layer_num + 1
        self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)], self.params['b'+str(idx)])
        self.last_layer = SoftmaxWithLoss()
    def __init_weight(self, weight_init_std):
        """设置权重的初始值"""
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx-1]) #使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid','xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx-1])
            self.params['W'+str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])
    def predict(self, x):
        # 各层的正向传播：推理【推理过程不需要softmax层,正向传播至最后一层Affine】
        for layer in self.layers.values():
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
        y = self.predict(x)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num+2):
            W = self.params['W'+str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        return self.last_layer.forward(y, t) + weight_decay
    def numerical_gradient(self, x, t):
        """数值微分法求梯度"""
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W'+str(idx)] = numerical_gradient_2(loss_W, self.params['W'+str(idx)])
            grads['b'+str(idx)] = numerical_gradient_2(loss_W, self.params['b'+str(idx)])
        return grads
    def gradient(self, x, t):
        """误差反向传播法求梯度"""
        #forward:最后softmax-with-loss层损失函数、传递梯度
        self.loss(x, t)
        #backward
        dout = 1
        dout = self.last_layer.backward(dout) #梯度，准备反向传播到下一层
        # 前面Affine层、激活层损失函数、传递梯度
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 设定：取出backward过程储存在Affine的权重参数、偏置参数的梯度
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W'+str(idx)] = self.layers['Affine'+str(idx)].dW +  self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b'+str(idx)] = self.layers['Affine'+str(idx)].db
        return grads
