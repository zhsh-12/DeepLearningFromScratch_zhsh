import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 创建综合示例数据
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 70000, 80000, 90000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'experience': ['Low', 'Medium', 'Medium', 'High', 'Low']
})

print("原始数据：")
print(data)

# 定义预处理管道
numeric_features = ['age', 'salary']
categorical_features = ['department', 'experience']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 应用预处理
processed_data = preprocessor.fit_transform(data)

print("\n预处理后的数据形状：", processed_data.shape)
print("\n预处理后的特征名称：")
feature_names = (numeric_features +
                 list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
print(feature_names)