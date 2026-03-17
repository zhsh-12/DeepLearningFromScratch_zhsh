import numpy as np
from sklearn.decomposition import PCA

# 生成示例数据
np.random.seed(42)
X = np.random.randn(100, 3)  # 100个样本，3个特征
X = X @ np.random.randn(3, 3)  # 引入相关性

# PCA白化
pca = PCA(whiten=True)
X_whitened = pca.fit_transform(X)

print("原始数据的协方差矩阵：")
print(np.cov(X.T))
print("\n白化后数据的协方差矩阵：")
print(np.cov(X_whitened.T))

# 验证结果
print(f"\n白化后数据的均值：{np.mean(X_whitened, axis=0)}")
print(f"白化后数据的标准差：{np.std(X_whitened, axis=0)}")