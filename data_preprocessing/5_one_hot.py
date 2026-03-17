import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 示例数据
data = pd.DataFrame({
    'color': ['红', '绿', '蓝', '绿', '红'],
    'size': ['S', 'M', 'L', 'M', 'S']
})

print("原始数据：")
print(data)

# 使用pandas的get_dummies
onehot_pd = pd.get_dummies(data, columns=['color', 'size'])
print("\n使用pandas的One-Hot编码：")
print(onehot_pd)

# 使用sklearn的OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(data[['color', 'size']])
encoded_df = pd.DataFrame(encoded_array,
                          columns=encoder.get_feature_names_out(['color', 'size']))
print("\n使用sklearn的One-Hot编码：")
print(encoded_df)