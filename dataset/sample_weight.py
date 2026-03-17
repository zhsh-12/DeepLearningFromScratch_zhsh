from pathlib import Path
import pickle

#第1步：获取当前文件绝对路径，项目根目录绝对路径
current_dir = Path(__file__).resolve()
base_dir = current_dir.parent.parent
# print(current_dir) #J:\PycharmProjects\DL-ZTKY\dataset\sample_weight.py
# print(base_dir) #J:\PycharmProjects\DL-ZTKY

#第2步：确认pkl文件绝对路径
pkl_path = base_dir / "data" / "sample_weight.pkl"
# print(pkl_path) #J:\PycharmProjects\DL-ZTKY\data\sample_weight.pkl

#第3步：加载pkl数据
def init_network():
    with open(pkl_path, 'rb') as f:
        network = pickle.load(f)
    return network
network = init_network()

#第4步：验证是否存在【3层神经网络：输入层784，隐藏层50，隐藏层100，输出层10】
# print(network['W1'].shape) #(784, 50)
# print(network['W2'].shape) #(50, 100)
# print(network['W3'].shape) #(100, 10)
# print(network['W1'][0])