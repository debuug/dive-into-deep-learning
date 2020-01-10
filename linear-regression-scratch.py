#coding=utf-8
'''
licensed to huwei
created on 12/27/2019
'''
from mxnet import nd,autograd
import random

# **********************生成数据集**********************
num_features = 2    # 特征数
num_samples = 1000 # 样本数
true_w = [2, -3.4]  # 真实权重
true_b = 4.2        # 真实偏差
samples = nd.random.normal(scale = 1, shape = (num_samples, num_features)) # 随机正态分布生成样本特征
lables = true_w[0] * samples[:, 0] + true_w[1] * samples[:, 1] + true_b     # 生成训练集的标签
lables += nd.random.normal(scale = 0.01, shape = lables.shape)  # 为标签加上标准差为0.01的随机正态分布噪声

# **********************读取数据函数**********************
# 从样本中随机抽取批量为batch_size的样本，直至完全抽取
# 例如样本为1000，小批量的大小为10，则每次随机抽取10个样本
# 一轮完整的抽取需要100次
def data_iter(batch_size, samples, lables):
    num_samples = len(samples)  # 计算样本数量
    indices = list(range(num_samples))  # 生成0~num_samples的list
    random.shuffle(indices) # 随机打乱indices
    for i in range(0, num_samples, batch_size):
        j = nd.array(indices[i : min(i + batch_size, num_samples)]) # 随机返回10个样本的下标数组
        yield samples.take(j), lables.take(j)   # take函数根据索引数组获取samples的数组子集
    
# **********************初始化模型参数**********************
# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0
w = nd.random.normal(scale = 0.01, shape = (num_features, 1))
b = nd.zeros(shape = (1,))
# 之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此需要创建它们的梯度
w.attach_grad()
b.attach_grad()

# **********************定义模型**********************
# X为样本，w为权重，b为偏差
def linear_reg(X, w, b):
    return nd.dot(X, w) + b

# **********************定义损失函数**********************
# pred_y为y的预测值，true_y为y的真实值
# 需要把真实值true_y变形成预测值pred_y的形状
# 损失函数除以2是为了使对平方项求导后的常数系数为1，形式上更加简洁
def squared_loss(pred_y, true_y):
    return (pred_y - true_y.reshape(pred_y.shape)) ** 2 / 2

# **********************定义优化算法**********************
# 由于param并非标量，MXNet自动求梯度时会对param中元素求和得到新的变量，
# 再求该变量对模型参数的梯度，相当于对batch_size个标量表达式求梯度，再求和
# 最后除以batch_size得到梯度平均值
def sgd(params, learn_rate, batch_size):
    for param in params:
        param[:] = param - learn_rate * param.grad / batch_size

# **********************训练模型**********************
batch_size = 10 # 一次抽取的样本数量
learn_rate = 0.03   # 学习率（超参数）
num_iter_cycle = 3  # 迭代周期个数（超参数），每个迭代周期会训练整个训练集
for iter_cycle in range(num_iter_cycle):
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。
    # X和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, samples, lables):
        with autograd.record():
            loss = squared_loss(linear_reg(X, w, b), y) # loss是有关小批量X和y的损失
        loss.backward() # 小批量的损失对模型参数求梯度
        sgd([w, b], learn_rate, batch_size) # 使用小批量随机梯度下降迭代模型参数
    # 完成一个迭代周期，训练集中所有样本均经过训练
    cur_cycle_loss = squared_loss(linear_reg(samples, w, b), lables)    # 计算整个训练集的损失
    print("Cycle %d, loss = %f" % (iter_cycle + 1, cur_cycle_loss.mean().asscalar()))
