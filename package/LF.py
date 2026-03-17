import numpy as np

#均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

# t = [0,0,1,0,0,0,0,0,0,0] #列表
# y = [0.1,0.05,0.6,0,0.05,0.1,0,0.1,0,0] #列表
# t_arr = np.array(t) #一维数组：[0 0 1 0 0 0 0 0 0 0]
# y_arr = np.array(y) #一维数组：[0.1  0.05 0.6  0.   0.05 0.1  0.   0.1  0.   0.  ]
# print(mean_squared_error(y_arr, t_arr)) #0.09750000000000003

#交叉熵误差
def cross_entropy_error_1(y, t):
    delta = 1e-7 #添加微小值，避免出现np.log(0)时，出现负无穷值
    return -np.sum(t * np.log(y + delta))
# print(cross_entropy_error_1(y_arr, t_arr)) #0.510825457099338

#mini-batch版交叉熵误差：同时处理单个数据和批量数据；one-hot标签
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size) #转为二维数组：[[0.1  0.05 0.6  0.   0.05 0.1  0.   0.1  0.   0.  ]]
        t = t.reshape(1, t.size) #转为二维数组：[[0 0 1 0 0 0 0 0 0 0]]
    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size
# print(cross_entropy_error(y_arr, t_arr)) #0.510825457099338

#mini-batch版交叉熵误差：同时处理单个数据和批量数据；标签形式（类似“2”，“7”这样的标签）
def cross_entropy_error_2(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
# t = [2] #1个样本数据对应的标签，列表形式
# y = [0.1,0.05,0.6,0,0.05,0.1,0,0.1,0,0] #1个样本数据，列表形式
# t_arr = np.array(t) #一维数组：[2]
# y_arr = np.array(y) #一维数组：[0.1  0.05 0.6  0.   0.05 0.1  0.   0.1  0.   0.  ]
# print(cross_entropy_error_2(y_arr, t_arr)) #0.510825457099338

#mini-batch版交叉熵误差：批量数据，输入时未取得标签索引（非整数形式）
def cross_entropy_error_3(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# y = [0.1,0.05,0.6,0,0.05,0.1,0,0.1,0,0]
# t = [0,0,1,0,0,0,0,0,0,0]
# print(cross_entropy_error_3(np.array(y), np.array(t))) #0.510825457099338

