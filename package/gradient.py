import numpy as np

#数值微分：中心差分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)
# def function(x):
#     return 0.01*x**2 + 0.1*x
# x = 5
# print(numerical_diff(function, x)) #这里必须传入函数function，0.1999999999990898

#梯度：参数f为函数,x为numpy数组，实现对numpy数组内各个元素求数值微分
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) #生成和x形状相同，所有元素为0的数组：（3，2）
    for idx in range(x.size): #x.size: 各维度乘积
        tmp_val = x[idx]
        #f(x+h)的计算
        x[idx] = float(tmp_val) + h #这里会产生浮点数截断问题
        fxh1 = f(x) #将整个数组传入函数f
        #f(x-h)的计算
        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad
# def function(x):
#     return np.sum(x**2)
# x = np.array([3.0,4.0]) #这里必须是浮点数，因为h是浮点数，为实现两者的相加减而不产生浮点数截断问题
# print(numerical_gradient(function, x)) #[6. 8.]

def numerical_gradient_2(f, x):
    h = 1e-4
    grad = np.zeros_like(x) #生成和x形状相同，所有元素为0的数组：（3，2）
    for idx in range(x.size): #x.size: 各维度乘积
        tmp_val = x.flat[idx] #展平数组后，依次取出各个元素
        #f(x+h)的计算
        x.flat[idx] = float(tmp_val) + h#为避免产生浮点数截断问题，调整为浮点数
        fxh1 = f(x) #将整个数组传入函数f
        #f(x-h)的计算
        x.flat[idx] = float(tmp_val) - h #为避免产生浮点数截断问题，调整为浮点数
        fxh2 = f(x)

        grad.flat[idx] = (fxh1 - fxh2) / (2*h)
        x.flat[idx] = tmp_val #还原值
    return grad

def numerical_gradient_3(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad


# def function(x):
#     return np.sum(x**2)
# x = np.array([[3.0,4.0],[2.0,0.0],[4.0,0.0]]) #这里必须是浮点数，因为h是浮点数，为实现两者的相加减而不产生浮点数截断问题
# print(numerical_gradient_2(function, x)) #[[6. 8.], [4. 0.], [8. 0.]]

#引入学习率，求函数的极小值(顺利的话，求函数的最小值）：数值微分法求梯度
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x) #梯度
        x -= lr * grad
    return x
# def function(x):
#     return x[0]**2 + x[1]**2
# init_x = np.array([-3.0,4.0])
# print(gradient_descent(function, init_x=init_x,lr=0.1, step_num=100)) #[-6.11110793e-10  8.14814391e-10]


