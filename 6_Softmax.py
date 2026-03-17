#softmax计算溢出问题
import numpy as np

#问题复现
a = np.array([1010,1000,990])
exp_a = np.exp(a) #RuntimeWarning: overflow encountered in exp
sum_exp_a = np.sum(exp_a)
y = exp_a/sum_exp_a #RuntimeWarning: invalid value encountered in divide
# print(y)

#处理方法
c = np.max(a) #最大的输入信号
print(c) #1010
new_a = a - c #减去输入信号中的最大值
print(new_a) #[  0 -10 -20]
new_y = np.exp(new_a)/np.sum(np.exp(new_a))
print(new_y) #[9.99954600e-01 4.53978686e-05 2.06106005e-09]

#模块引用
from package.NN import softmax
a = np.array([1010,1000,990])
y = softmax(a)
print(y) #[9.99954600e-01 4.53978686e-05 2.06106005e-09]