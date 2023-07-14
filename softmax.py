import numpy as np
import torch
import torch.nn.functional as F


def softmax(a):
    a -= np.max(a)
    return np.exp(a) / np.sum(np.exp(a))


a = input("")
if " " in a:
    arr = a.split(" ")
    for i in range(0,len(arr)):
        arr[i] = eval(arr[i])
else:
    arr = a.split(",")
    for i in range(0,len(arr)):
        arr[i] = eval(arr[i])
n = np.array(arr)
print(n.shape)
data = torch.FloatTensor(arr)
print(softmax(n))
prob = F.softmax(data, dim=0)  # dim = 0,在列上进行Softmax;dim=1,在行上进行Softmax
print(prob)
print(prob.shape)
print(prob.type())

