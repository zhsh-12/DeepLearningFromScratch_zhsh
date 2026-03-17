import numpy as np
from collections import OrderedDict
from package.layer import Sigmoid, Relu, Affine, SoftmaxWithLoss, BatchNormalization, Dropout
from package.gradient import numerical_gradient_2

class MultiLayerNetExtend:
    """扩展版的全连接的多层神经网络：具有weight decay、dropout、batch normalization功能
    input_size：输入大小（MNIST情况下为784）
    hidden_size_list：隐藏层的神经元数量列表（e.g. [100,100,100,100]）
    output_size: 输出大小（MNIST情况下为10）
    activation: 'relu' 或 'sigmoid'
    weight_init_std: 指定权重的标准差（e.g. 0.01）
       指定'relu'或'He'情况下设定‘He的初始值’
       指定'sigmoid'或‘Xavier’情况下设定'Xavier的初始值’
    weight_decay_lambda：weight decay（L2范数）的强度
    use_dropout: 是否使用dropout
    dropout_ration: dropout的比例
    use_batchNorm: 是否使用Batch Normalization
    """
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu',
                 weight_init_std='relu', weight_decay_lambda=0, use_dropout=False,
                 dropout_ration=0.5, use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}
        
        #初始化权重
        self.__init_weight(weight_init_std)
       
        ##构造层：加权层+激活层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)],self.params['b'+str(idx)])
            if self.use_batchnorm:
                self.params['gamma'+str(idx)] = np.ones(hidden_size_list[idx-1]) #gamma一开始为1
                self.params['beta'+str(idx)] = np.zeros(hidden_size_list[idx-1]) #beta一开始为0
                self.layers['BatchNorm'+str(idx)] = BatchNormalization(self.params['gamma'+str(idx)], self.params['beta'+str(idx)])
            self.layers['Activation_function'+str(idx)] = activation_layer[activation]() #创造激活层实例
            if self.use_dropout:
                self.layers['Dropout'+str(idx)] = Dropout(dropout_ration)
        idx = self.hidden_layer_num + 1
        self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)], self.params['b'+str(idx)])
        self.last_layer = SoftmaxWithLoss()
    
    def __init_weight(self, weight_init_std):
        """设定权重的初始值"""
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu','he'):
                scale = np.sqrt(2.0 / all_size_list[idx-1]) #使用Relu情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid','xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx-1]) #使用sigmoid情况下推荐的初始值
            self.params['W'+str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])
    
    def predict(self, x, train_flg=False):
        """预测值"""
        for key, layer in self.layers.items():
            if 'Dropout' in key or 'BatchNorm' in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x
    
    def loss(self, x, t, train_flg=False):
        """损失函数"""
        y = self.predict(x, train_flg)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num+2):
            W = self.params['W'+str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        return self.last_layer.forward(y, t) + weight_decay
    
    
    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim !=1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        """求梯度（数值微分法）"""
        loss_W = lambda W: self.loss(x, t, train_flg=True)
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W'+str(idx)] = numerical_gradient_2(loss_W, self.params['W'+str(idx)])
            grads['b'+str(idx)] = numerical_gradient_2(loss_W, self.params['b'+str(idx)])
            if self.use_batchnorm and idx != self.hidden_layer_num+1: #最后一层没有Batch Norm处理
                grads['gamma'+str(idx)] = numerical_gradient_2(loss_W, self.params['gamma'+str(idx)])
                grads['beta'+str(idx)] = numerical_gradient_2(loss_W, self.params['beta'+str(idx)])
        return grads
    
    def gradient(self, x, t):
        """求梯度（误差反向传播法）"""
        self.loss(x, t, train_flg=True)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W'+str(idx)] = self.layers['Affine'+str(idx)].dW + self.weight_decay_lambda*self.params['W'+str(idx)]
            grads['b'+str(idx)] = self.layers['Affine'+str(idx)].db
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta
        return grads


