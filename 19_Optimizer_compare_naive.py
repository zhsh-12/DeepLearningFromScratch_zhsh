import matplotlib.pyplot as plt
from collections import OrderedDict
from package.optimizer import *

#目标函数
def f(x,y):
    return x**2/20.0 + y**2
#梯度函数
def df(x,y):
    return x/10.0, 2.0*y

#初始化设置：初始位置、参数字典、梯度字典
init_pos = (-7.0, 2.0)
params = {}
grads = {}

#定义优化器：使用有序字典保持顺序
optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr=0.95)
optimizers['Momentum'] = Momentum(lr=0.1)
optimizers['AdaGrad'] = AdaGrad(lr=1.5)
optimizers['Adam'] = Adam(lr=0.3)

idx = 1
for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]

    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params,grads)

    #创建等高线图数据
    x = np.arange(-10,10, 0.01)
    y = np.arange(-5,5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    mask = Z > 7
    Z[mask] = 0 #限制等高线显示范围，只显示Z <= 7区域，使图像更清晰

    #plot绘图
    plt.subplot(2,2,idx) #子图的规划位置：2行2列左上第1位置
    idx += 1
    plt.plot(x_history, y_history, 'o-', color='red') #绘制优化轨迹，数据点用红色圆圈标记，默认实线连接
    plt.contour(X, Y, Z) #在X，Y网格点上绘制Z值的等高线
    plt.ylim(-10,10)
    plt.xlim(-10,10)
    plt.plot(0,0,"+") #用“+”标记原点
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
plt.tight_layout() #调整4个子图的布局，避免重叠
plt.show()