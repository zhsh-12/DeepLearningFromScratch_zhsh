import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

# 示例数据
data = np.array([[10, 20], [30, 40], [50, 60]])

# 最小-最大正规化
min_max_scaler = MinMaxScaler()
normalized_minmax = min_max_scaler.fit_transform(data)
print("最小-最大正规化结果：")
print(normalized_minmax)

# 最大绝对值正规化
max_abs_scaler = MaxAbsScaler()
normalized_maxabs = max_abs_scaler.fit_transform(data)
print("\n最大绝对值正规化结果：")
print(normalized_maxabs)