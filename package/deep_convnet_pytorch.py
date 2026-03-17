import torch.nn as nn
import torch.nn.functional as F

class DeepConvNetPyTorch(nn.Module):
    """使用Pytorch重构DeepConvNet:
        conv - relu - pool -
        conv - relu - pool -
        affine - relu - dropout - 
        affine - dropout - softmax"""
    def __init__(self, dropout_ratio=0.5):
        super(DeepConvNetPyTorch, self).__init__()
        #定义层：所有参数由Pytorch自动初始化并管理
        #2次卷积
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1)
        #2次池化(2*2, stride=2)
        #2次全连接
        self.fc1 = nn.Linear(64*7*7, 50)
        self.fc2 = nn.Linear(50, 10)
        #2次Dropout
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)
        
        #初始化权重
        self.__initialize_weights() #自定义方法，私有方法【权重、偏置】
    
    def __initialize_weights(self):
        for m in self.modules(): #遍历模型中的所有生成层
            if isinstance(m, (nn.Conv2d, nn.Linear)): #检查当前模块是否是卷积层，或全连接层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #kaiming初始化/He初始化方法，要初始化的权重张量，基于该层的输出神经元数量【fan_in基于输入维度，前者在某些情况下可能带来更好的训练稳定性】
                if m.bias is not None: #偏置
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x, train_flg=False):
        """前向传播
        Args: 
            x: 输入张量[batch_size, 1, 28, 28]
            train_flg: 是否为训练模式(影响Dropout行为)
        Returns:
            输出张量[batch_size, 10]
        """
        #第一个卷积块
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) #池化窗口2*2，步长2
        #第二个卷积块
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        #展开以输入全连接层
        x = x.view(x.size(0), -1) #[batch_size, 64*7*7]
        #第一个全连接块（仿射变换）
        x = F.relu(self.fc1(x))
        x = self.dropout1(x) if train_flg else x   #仅在训练时启用Dropout
        #第二个全连接块（输出层）
        x = self.fc2(x)
        x = self.dropout2(x) if train_flg else x 
        return x
    
    #为保持接口兼容，可以添加loss方法【可选】
    def loss(self, x, t, train_flg=True):
        y = self.forward(x, train_flg)
        criterion = nn.CrossEntropyLoss()
        return criterion(y, t)
        
