import numpy as np
import sys
sys.path.append(".") # 将项目根目录加入路径，以便导入dataset模块
from dataset.mnist import load_mnist
from dataset.sample_weight import init_network
from package.NN import sigmoid, softmax
#第1步：导入mnist数据集：测试集图像文件、标签文件
def get_data():
    x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=True,one_hot_label=False)
    return x_test, t_test
x, t = get_data()
#第2步：初始化权重、偏置等参数，保存在字典变量network
network = init_network()
#第3步：构建神经网络，进行推理预测
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    A1 = np.dot(x, W1) + b1
    z1 = sigmoid(A1)
    A2 = np.dot(z1, W2) + b2
    z2 = sigmoid(A2)
    A3 = np.dot(z2, W3) + b3
    y = softmax(A3)
    return y
#第4步：【批处理】评价上述神经网络的预测正确率accuracy
batch_size = 100 #批数量
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) #获取概率最高的元素的索引，沿着第1维方向（即：列方向）
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) #比较运算符生成布尔型数组，计算True的个数
print(f"Accuracy: {accuracy_cnt/len(x)}") #Accuracy: 0.9352




