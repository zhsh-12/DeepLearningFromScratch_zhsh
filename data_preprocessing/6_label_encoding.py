from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 示例数据
data = pd.DataFrame({
    'animal': ['猫', '狗', '鸟', '狗', '猫', '鸟'],
    'size': ['小', '中', '小', '大', '中', '小']
})

print("原始数据：")
print(data)

# 对动物类型进行标签编码
le_animal = LabelEncoder()
data['animal_encoded'] = le_animal.fit_transform(data['animal'])

# 对大小进行标签编码（有序）
le_size = LabelEncoder()
size_mapping = {'小': 0, '中': 1, '大': 2}
data['size_encoded'] = data['size'].map(size_mapping)

print("\n编码后的数据：")
print(data)

print(f"\n动物类别映射：")
for i, class_ in enumerate(le_animal.classes_):
    print(f"{class_}: {i}")