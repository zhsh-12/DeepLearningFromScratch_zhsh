import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    """创建基于Pytorch的神经网络训练类"""
    def __init__(self, network, device, x_train, t_train, x_test, t_test, batch_size=100, max_epochs=20, verbose=True, eval_batch_size=None):
        self.device = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        
        #修正1：使用.copy()确保获得numpy数组的可写副本，此时数据在CPU上
        x_train = torch.from_numpy(x_train.copy()).float()
        t_train = torch.from_numpy(t_train.copy()).long() # 分类标签用long类型
        x_test = torch.from_numpy(x_test.copy()).float()
        t_test = torch.from_numpy(t_test.copy()).long()
        
        #创建数据集和DataLoader【数据在CPU上】
        self.train_dataset = TensorDataset(x_train, t_train)
        self.test_dataset = TensorDataset(x_test, t_test)
        
        #修正2：pin_memory是为CPU数据加速到GPU传输的，这里保留True，windows系统设置多进程为0，pin_memory仅在数据位于CPU时有效
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0) 
        self.eval_train_loader = DataLoader(self.train_dataset, batch_size=self.eval_batch_size, shuffle=False, pin_memory=True, num_workers=0)
        self.eval_test_loader = DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False, pin_memory=True, num_workers=0)
              
        #初始化模型、优化器、损失函数
        self.model = network.to(self.device) #模型移到GPU
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) #学习率调整：0.001 - 0.01 -0.1
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc_history = []
        self.test_acc_history = []
        self.train_loss_history = []
    
    def __eval_accuracy(self, data_loader):
        #私有方法，评定训练集、测试集的准确率
        self.model.eval() #开启评估阶段
        correct = 0.0
        total = 0.0
        with torch.no_grad(): #不计算梯度，节省内存和计算
            for batch_x, batch_t in data_loader:
                batch_x, batch_t = batch_x.to(self.device), batch_t.to(self.device) #将批次数据移至设备GPU
                outputs = self.model(batch_x, train_flg=False) #前向传播
                _, predicted = torch.max(outputs.data, 1) #获取预测结果
                total += batch_t.size(0)
                correct += (predicted == batch_t).sum().item()
        return correct / total if total > 0.0 else 0.0
    
    def train(self):
        #主训练方法
        for epoch in range(self.max_epochs):
            #开启训练阶段(启用Dropout等)
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_t in self.train_loader:
                #修正3：每个批次训练时，将数据从CPU移动指定的设备GPU
                batch_x, batch_t = batch_x.to(self.device), batch_t.to(self.device)
                self.optimizer.zero_grad() #清零梯度
                outputs = self.model(batch_x, train_flg=True) #前向传播
                loss = self.criterion(outputs, batch_t) #计算损失
                loss.backward() #反向传播
                self.optimizer.step() #更新参数
                running_loss += loss.item() * batch_x.size(0) #计算当前批次损失  
            epoch_loss = running_loss / len(self.train_dataset) #计算当前epoch损失
            self.train_loss_history.append(epoch_loss)
            if self.verbose: print("train_epoch_loss: " + str(epoch_loss))
            
            #评估阶段
            train_acc = self.__eval_accuracy(self.eval_train_loader)
            test_acc = self.__eval_accuracy(self.eval_test_loader)
            self.train_acc_history.append(train_acc)
            self.test_acc_history.append(test_acc)
            print(f'Epoch {epoch+1:2d}/{self.max_epochs} | Loss: {epoch_loss:.4f} |Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')
    
    def save_model(self, file_name='deep_convent_pytorch.pth'):
        torch.save(self.model.state_dict(), 'deep_convnet_pytorch.pth')
    
    
       
       
        
       
       
    
    
    
   
       
       
        
        
        
      
    
   
    
    
    
    
    
    