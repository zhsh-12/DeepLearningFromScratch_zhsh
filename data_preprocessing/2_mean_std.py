from sklearn.preprocessing import StandardScaler
import numpy as np

# 示例数据
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# 标准化
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

print("原始数据：")
print(data)
print(f"\n均值：{scaler.mean_}")
print(f"标准差：{scaler.scale_}")
print("\n标准化后数据：")
print(standardized_data)