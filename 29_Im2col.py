import sys
sys.path.append(".")
import numpy as np
from package.util import im2col

x1 = np.random.rand(1,3,7,7) #批大小为1，通道为3的7*7数据
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape) #(9, 75)

x2 = np.random.rand(10,3,7,7) #批大小为10，通道为3的7*7数据
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) #(90, 75)
