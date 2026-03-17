import numpy as np
import pickle
import gzip
import os
import urllib.request #前提：使用python 3.X版本

# 定义pickle文件的保存位置，与原始MNIST数据在相同目录下：./data
save_file = './data/mnist.pkl'

# 注意：PyTorch默认会下载并解压.gz文件，所以我们要找解压后的 .ubyte 文件
raw_dir = './data/MNIST/raw/'
train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)  # 通道, 高, 宽
img_size = 784  # 28*28

#文件下载：如果本地没有原始文件，则从网络下载（备用）
def _download(file_name):
    file_path = os.path.join(raw_dir, file_name)
    if os.path.exists(file_path):
        print(f"{file_name} already exists.")
        return

    print(f"Downloading {file_name} ... ")
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + file_name, file_path)
    print("Done")

#文件是否存在【不存在启动文件下载】：确保所有原始文件存在（如果PyTorch已下载，则跳过）
def download_mnist():
    os.makedirs(raw_dir, exist_ok=True)
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    for file in files:
        gz_path = os.path.join(raw_dir, file)
        # 检查.gz文件是否存在
        if not os.path.exists(gz_path):
            # 如果连解压后的文件也没有，则尝试下载.gz
            ubyte_file = file.replace('.gz', '')
            ubyte_path = os.path.join(raw_dir, ubyte_file)
            if not os.path.exists(ubyte_path):
                _download(file) #下载文件函数
            else:
                print(f"Found uncompressed file: {ubyte_file}, skipping download of {file}.")
        else:
            print(f"Found existing file: {gz_path}")

#加载标签文件【先检查文件是否存在】
def _load_label(file_name):
    file_path = os.path.join(raw_dir, file_name)
    print(f"Converting {file_name} to NumPy Array ...")
    # 尝试直接打开解压后的文件
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    else:
        # 如果不存在，尝试打开.gz文件
        gz_path = file_path + '.gz'
        if not os.path.exists(gz_path):
            download_mnist()  # 确保文件已下载
        with gzip.open(gz_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    return labels

#加载图像文件【先检查文件是否存在】
def _load_img(file_name):
    file_path = os.path.join(raw_dir, file_name)
    print(f"Converting {file_name} to NumPy Array ...")
    # 尝试直接打开解压后的文件
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    else:
        # 如果不存在，尝试打开.gz文件
        gz_path = file_path + '.gz'
        if not os.path.exists(gz_path):
            download_mnist()  # 确保文件已下载
        with gzip.open(gz_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

    # 数据形状转换： (num, 1, 28, 28) -> (num, 784)，与书中一致
    data = data.reshape(-1, img_size) #img_size = 784
    print("Done")
    return data

#原始数据文件 -> 4个numpy数组【加载图像文件、加载标签文件】
def _convert_numpy():
    # 分别加载四个部分
    train_img = _load_img('train-images-idx3-ubyte')
    train_label = _load_label('train-labels-idx1-ubyte')
    test_img = _load_img('t10k-images-idx3-ubyte')
    test_label = _load_label('t10k-labels-idx1-ubyte')
    # 返回格式与书中完全相同
    return train_img, train_label, test_img, test_label

#标签转换为one-hot格式（可选，原书函数包含此功能）
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    T[np.arange(X.size), X] = 1  # T[行索引[0,1,2,...]，列索引[X的标签]]
    return T

#初始化MNIST数据集：解析二进制文件并保存为pickle【先检查文件是否存在、再进行数据转换】
def init_mnist():
    download_mnist()  # 确保原始文件存在
    dataset = _convert_numpy()  # 解析为NumPy数组

    # 保存为pickle文件
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    print(f"Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)  # -1表示使用最高协议等级
    print(f"Done! Pickle file saved at: {save_file}")
    return dataset

#加载.pkl文件，获取所有数据【不存在则先初始化.pkl文件】
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """加载MNIST数据集（主函数，与原书接口一致）
    参数:
        normalize: 是否将图像的像素值标准化到0.0~1.0（默认为True）
        flatten: 是否将图像展开为一维数组（默认为True）
        one_hot_label: 是否将标签转换为one-hot格式（默认为False）
    返回:
        (训练图像, 训练标签, 测试图像, 测试标签)
    """
    # 如果pickle文件不存在，则初始化创建
    if not os.path.exists(save_file):
        init_mnist()

    # 从pickle文件加载
    # print(f"Loading pickle file from: {save_file}")
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    # 解包数据
    train_img, train_label, test_img, test_label = dataset

    # 根据参数进行数据处理（与书中逻辑完全一致）
    if normalize:
        train_img = train_img.astype(np.float32) / 255.0
        test_img = test_img.astype(np.float32) / 255.0

    if not flatten:
        # 如果不要扁平化，则转为(样本数, 通道, 高, 宽)格式
        train_img = train_img.reshape(-1, 1, 28, 28)
        test_img = test_img.reshape(-1, 1, 28, 28)

    if one_hot_label:
        train_label = _change_one_hot_label(train_label)
        test_label = _change_one_hot_label(test_label)

    # 返回格式与书中完全相同
    return train_img, train_label, test_img, test_label

