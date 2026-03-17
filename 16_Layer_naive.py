#乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
#加法层
class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
#实例化
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1
#layer
mul_apple_layer = MulLayer() #apple, apple_num
mul_orange_layer = MulLayer() #orange, orange_num
add_apple_orange_layer = AddLayer() #apple_price, orange_price
mul_tax_layer = MulLayer() #all_price, tax
#forward
apple_price = mul_apple_layer.forward(apple, apple_num) #1
orange_price = mul_orange_layer.forward(orange, orange_num) #2
all_price = add_apple_orange_layer.forward(apple_price,orange_price)#3
price = mul_tax_layer.forward(all_price, tax)#4
#backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice) #4
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)#3
dorange, dorange_num = mul_orange_layer.backward(dorange_price)#2
dapple, dapple_num = mul_apple_layer.backward(dapple_price)#1

print(int(price)) #715
print(dapple_num,dapple, dorange,dorange_num,dtax) #110.00000000000001 2.2 3.3000000000000003 165.0 650