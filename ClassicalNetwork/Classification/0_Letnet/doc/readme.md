





conda activate yolov5-6.2

20230905:
    watch -n 1 -c gpustat --color 
    



Reference:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html



https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html



N=(W-F+2P)/S+1
输入图片大小 w x w
Filter大小 F X F 
步长S
padding的像素数P



- 卷积核的channel与输入特征层的channel相同
- 输出的特征矩阵的channel与卷积核个数相同


思考?
加上偏移量bias该如何计算?
    只需要在输出层的每一个元素上加上bias的值即可。

加上激活函数该如何计算?
    

如果卷积过程中出现越界的情况该怎么办?
    填充0



# 池化层
- 没有训练参数只改变特征矩阵的w和h
- 不改变channel
- 一般poolsize和stride相同


# 在pytorch官网查找算子的信息
i.e.https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d

 # Reference
 https://blog.csdn.net/m0_37867091/article/details/107136477
 