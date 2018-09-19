---
layout:     post
title:      Python
subtitle:   numpy && pandas
date:       2018-09-19
author:     ZYC
header-img: img/post-bg-ios9-web.jpg
catalog: 	 true
tags:
    - deeplearing
    - python
---

### install

pip3 install numpy
pip3 install pandas

测试安装:

>import pandas as pd
pd.test()

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
array = np.ones(...)
np.arange(begin,end,step) 
reshape((x1,x2,...,xn)) #重新变化数组的shape,x1*x2*...*xn=原来的长度
np.full((2,2), 7)  # Create a constant array
np.eye(2)         # Create a 2x2 identity matrix，创建一个单位矩阵
np.random.random((2,2))  # Create an array filled with random values

#### numpy基础元算

