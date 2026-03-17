import numpy as np

# 示例数据
data = np.array([10, 20, 30, 40, 50])

# 计算均值
mean_val = np.mean(data)

# 移动数据（数据中心化）
shifted_data = data - mean_val

print(f"原始数据：{data}")
print(f"均值：{mean_val}")
print(f"移动后数据：{shifted_data}")
print(f"移动后数据的均值：{np.mean(shifted_data)}")