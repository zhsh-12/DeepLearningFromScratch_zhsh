#阶跃函数
import numpy as np
import matplotlib.pyplot as plt
def step_function_1(x):
    y = x > 0 #数组的每个元素均进行不等号运算，生成一个布尔型数组
    return y.astype(int) #将布尔型数组转换为int类型
a = np.array([-1.9,0.6])
# print(step_function_1(a))  #[0 1]

def step_function(x):
    return np.array(x > 0, dtype=int)
# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

#sigmoid函数
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x)) #numpy的广播功能
# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1)
# plt.show()

#ReLU函数
import numpy as np
import matplotlib.pyplot as plt
def relu(x):
    return np.maximum(0, x) #选择最大值输出
# x = np.arange(-5.0, 5.0, 0.1)
# y = relu(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 5.1)
# plt.show()

