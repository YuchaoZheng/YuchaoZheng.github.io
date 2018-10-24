## Convolutional Neural Networks(deeplearning.ai)笔记

### Week 1

[参考博客1](https://blog.csdn.net/libbyandhelen/article/details/78747964)
[参考博客2](https://blog.csdn.net/red_stone1/article/details/78651063)

Computer Vision Porblems
- Image Classifcation
- Object detection
- Neural Style Transfer

$Input\ too\ large -> fc参数太多 -> 卷积(参数共享，稀疏连接)$

![](https://i.stack.imgur.com/SFST9.gif)

**Padding:**

普通的卷积运算会出现两个问题：

$n * n的图像与f*f的filter卷积会输出（n-f+1）*（n-f+1）的图像，这意味着$
  $图像每经过一次运算就会变小一点$
$在角落和边缘的像素被filter覆盖的次数没有处于中心的像素多，这说明图像边角的信息$ 
  $可能被忽略$

$在输入图像的四周加上宽度为p的边框，这样输入图像的大小变为  (n+2*p) *(n+2*p),$
$
输出（n+2*p-f+1）*（n+2*p-f+1）的图像
$

![](http://deeplearning.net/software/theano/_images/arbitrary_padding_no_strides.gif)

** 两种Padding方式**
- $Valid: no\ padding$
- $Same:Pad\ so\ that\ ouput\ size\ is\ the\ same\ as\ the\ input. -> p = (f-1)/2$

** filter 边长基本为奇数 **
1. $p = (f-1)/2，如果f为偶数，意味着对应维度前后的padding不同$
2. $p为奇数，有一个中心(p/2,p/2)$

** strided convolution**

![](https://cdn-images-1.medium.com/max/1600/0*1PSMTM8Brk0hsJuF.)

$n * n的图像与f*f的filter卷积,Padding为p会输出floor(((n-f+1+2*p)/s +1))*floor(((n-f+1+2*p)/s +1))的图像 $

**Convolutions Over Volume **

![](https://i.imgur.com/YJOMijb.png)
输入图像的通道数必须和filter的通道数一样，每次将27个数全部加起来，得到输出图像中的一个像素 

> 多个卷积:每一个卷积核有不同的含义，例如第一个卷积核水平边缘检测，第二个卷积核垂直边缘检测。

![多个filter](https://camo.githubusercontent.com/a330ed771274d0e0876333b1c42c7718f34d62c9/68747470733a2f2f692e696d6775722e636f6d2f5555476a745a5a2e706e67)

** One Layer of a Convolutional Network**

> bias的shape应该为[1,channels],每一个filter对应一个bi

![](https://raw.githubusercontent.com/YuchaoZheng/MarkdownPhotos/master/Res/20171128233242494.jpg)

>在CNN中，参数数目只由滤波器组决定。参数数目与输入图像的大小的无关的。

![](https://raw.githubusercontent.com/YuchaoZheng/MarkdownPhotos/master/Res/015.jpg)

** Types of layer in a convolution network:**
- Convolution
- Pooling
- Fully connected

** Pooling **
背后的机制：
只保留区域内的最大值（特征），忽略其它值，降低noise影响，提高模型健壮性。

如果是多个通道，那么就每个通道单独进行max pooling操作。

** CNN example**

![](https://raw.githubusercontent.com/YuchaoZheng/MarkdownPhotos/master/Res/016.jpg)

** Why convolution**

- 参数共享:一个特征检测器（例如垂直边缘检测）对图片某块区域有用，同时也可能作用在图片其它区域。
- 连接的稀疏性:因为滤波器算子尺寸限制，每一层的每个输出只与输入部分区域内有关。