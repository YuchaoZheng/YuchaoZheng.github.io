---
layout:     post
title:      Python
subtitle:   numpy && pandas
date:       2018-09-19
author:     ZYC
header-img: img/post-bg-ios9-web.jpg
catalog: 	 true
tags:
    - deeplearning
    - python
---

### install

pip3 install numpy
pip3 install pandas

测试安装:

> import pandas as pd
> pd.test()

### Numpy

基于矩阵的计算

创建一个numpy中的数组
> array = np.array([...])

array是一个几维
> array.ndim

array的每一维长度
> array.shape

array的size
> array.size

array可以指定type
>array = np.array(...,dtype=np.int64)

array快速初始化
> array = np.zeros((3,4),dtype=float)
> array = np.ones(...)
> np.arange(begin,end,step) 
> reshape((x1,x2,...,xn)) #重新变化数组的
> shape,x1*x2*...*xn=原来的长度
> np.full((2,2), 7)  # Create a constant array
> np.eye(2)         # Create a 2x2 identity matrix，创建一个单位矩阵
> np.random.random((2,2))  # Create an array filled with random values

#### numpy基础元算

numpy中两个矩阵的加减乘除对应相应位置的加减乘除。
每个位置的数平方用**
numpy中矩阵相乘np.dot(x,y)，不使用numpy,x.dot(y)


axis=0表示行相加，axis=1表示列相加(sum,max,min)
>x = np.array([[1,2],[3,4]])

>print(np.sum(x))  # Compute sum of all elements; prints "10"
>print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
>print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"

np.argmax,argmin找到最小值和最大值对应的位置。(一维)
np.mean求平均值
np.median求中位数
> np.cumsum累加和
> 例如A = np.arange(2,14).reshape((3,4))
> np.cumsum(A)为array([ 2,  5,  9, 14, 20, 27, 35, 44, 54, 65, 77, 90])

np.diff相邻两个数的差
np.sort排序，每行排序
np.transpose矩阵对称
np.clip(A, min_num, max_num),让A数组的值在min_num到max_num之间。

#### Numpy的索引

A数组的第二行第一列
A[2][1]或A[2,1]

A 是一个二维数组
> for row in A: 迭代A中的行
> for colume in A.T: 迭代A中的列

#### array的合并

只能合并两个
上下合并:np.vstack((A,B))
左右合并:np.hstack((A,B))

多个array的合并：np.consatenate((A1,A2,...),axis)

#### array的分割
np.splie()