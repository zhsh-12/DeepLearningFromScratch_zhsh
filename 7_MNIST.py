import torchvision
#获得训练集6万图片和对应的标签
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
print(f"数据集保存在：{train_dataset.raw_folder}") #数据集保存在：./data\MNIST\raw
#获得测试集1万图片和对应的标签
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
print(f"数据集保存在：{test_dataset.raw_folder}") #数据集保存在：./data\MNIST\raw