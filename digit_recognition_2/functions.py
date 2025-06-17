# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) 
        y = y.reshape(1, y.size) 
    
    """
    这段代码的功能是处理监督数据t的格式转换，具体含义如下：

1. 条件判断 ： if t.size == y.size
   
   - 检查监督数据t和预测值y的大小是否相同
   - 如果相同，说明t是以one-hot向量形式编码的监督数据
2. 格式转换 ： t = t.argmax(axis=1)
   
   - 将one-hot向量转换为对应的类别索引
   - 例如：one-hot向量[0,0,1,0] → 类别索引2
   - axis=1表示沿着每一行取最大值索引
3. 应用场景 ：
   
   - 在分类任务中，监督数据可能有两种形式：
     1. 直接类别标签（如数字2）
     2. one-hot编码（如[0,0,1,0]）
   - 这段代码确保无论输入哪种形式，都能统一转换为类别索引格式
4. 后续处理 ：
   
   - 转换后的t可以直接用于计算交叉熵误差
   - 例如： y[np.arange(batch_size), t] 可以正确选取每个样本对应类别的预测概率
    
    """
    if t.size == y.size:
        t = t.argmax(axis=1) 
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
