#将python对象序列化，保存至文件或字节流；反序列化，还原字节流数据为python对象
import pickle

data1 = [1,2,3]
data2 = {'a': 1, 'b': 2, 'c': 3}
data3 = "Hello World"

#序列化对象至文件
with open('data.pkl', 'wb') as f:
    pickle.dump(data1, f)
    pickle.dump(data2, f)
    pickle.dump(data3, f)
    f.close()
#反序列化还原字节流
with open('data.pkl', 'rb') as f:
    load1 = pickle.load(f)
    load2 = pickle.load(f)
    load3 = pickle.load(f)
    print(load1) #[1, 2, 3]
    print(load2) #{'a': 1, 'b': 2, 'c': 3}
    print(load3) #Hello World

#序列化对象为字节流
original_data = ['apple','banana', 123, {'key': 'value'}]
byte_stream = pickle.dumps(original_data)
print(f"字节流：{byte_stream}")
#字节流：b'\x80\x04\x95)\x00\x00\x00\x00\x00\x00\x00]\x94(\x8c\x05apple\x94\x8c\x06banana\x94K{}\x94\x8c\x03key\x94\x8c\x05value\x94se.'
#反序列化
restored_data = pickle.loads(byte_stream)
print(f"还原的数据：{restored_data}") #还原的数据：['apple', 'banana', 123, {'key': 'value'}]