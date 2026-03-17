
# Deep Learning from Scratch

源于斋藤康毅[著]，陆宇杰[译]《深度学习入门 基于Python的理论与实现》一书源码的实现【2026年更新版】。
代码过程含个人学习注释，欢迎批评指正。


# 文件说明

- data： 源数据集MNIST
- dataset： MNIST源数据的导入前处理
- data_preprocessing：【新增】常见的数据预处理方法【注：运行前需另外安装sklearn、pandas库】
- package：方便导入其他模块的文件，常见包括：激活函数、损失函数、梯度、神经网络设计、优化器、训练器等
- 根目录所有.py文件：可根据原书学习顺序依次运行，部分.py文件是过程拓展学习的，如：序号8、22、29、34、39等。
- 项目依赖：requirements.txt

本书指导我们由0到1构建深度学习网络，期间没有引入任何学习框架；为进行比对，特意在学习末尾引入Pytorch框架，实现了基于MNIST数据集的识别模型【序号39.py】。

# 环境说明

- 开发平台：windows[v11]，基础显卡[NVIDIA GPU]
- 开发环境：vscode[python 3.11]
- 项目依赖：先创设.venv，激活后，再按pip install -r requirements.txt安装所需依赖
          【注：torch、torchaudio、torchvision版本与个人电脑配置有关，需视情况安装；如果不打算引入Pytorch框架也可以不用安装】
- GPU加速：需先确认NVIDIA显卡的存在，且系统已安装CUDA和cuDNN【引入GPU加速的前提，只应用CPU计算可以不用考虑】


# 使用许可

本源代码使用MIT许可协议。 
