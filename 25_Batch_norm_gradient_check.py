import numpy as np
import sys
sys.path.append('.')
from dataset.mnist import load_mnist
from package.multi_layer_net_extend import MultiLayerNetExtend

x_train, t_train, x_test, t_test = load_mnist(normalize=True, flatten=True, one_hot_label=True)
x_batch = x_train[:1]
t_batch = t_train[:1]

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100], output_size=10, use_batchnorm=True)
grad_backprop = network.gradient(x_batch, t_batch)
grad_numerical = network.numerical_gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
# W1:0.0
# b1:0.0
# gamma1:0.0
# beta1:0.0
# gamma1:0.0
# beta1:0.0
# beta1:0.0
# W2:0.0
# b2:0.0
# W2:0.0
# b2:0.0
# gamma2:0.0
# beta2:0.054152715314326194
# b2:0.0
# gamma2:0.0
# beta2:0.054152715314326194
# gamma2:0.0
# beta2:0.054152715314326194
# W3:0.0
# beta2:0.054152715314326194
# W3:0.0
# W3:0.0
# b3:1.7990402263745597e-07
# b3:1.7990402263745597e-07