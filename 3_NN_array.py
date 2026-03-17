import numpy as np

A = np.array([1,2,3,4])
# print(A)
# print(type(A))
# print(A.shape)
# print(np.ndim(A))

B = np.array([[1,2],[3,4],[5,6]])
# print(B)
# print(B.shape)
# print(np.ndim(B))

A = np.array([[1,2],[3,4],[5,6]])
print(A.shape) #(3, 2)
B = np.array([7,8])
print(B.shape) #(2,)
result = np.dot(A,B)
print(result) #[23 53 83]
result = np.dot(B,A)
print(result) #报错：ValueError: shapes (2,) and (3,2) not aligned: 2 (dim 0) != 3 (dim 0)