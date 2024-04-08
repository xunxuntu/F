
# 如何训练自己的数据集
依据flower_photos文件的格式来设置你的数据集，然后将model.py中的num_classes修改一下即可。


sigmoid激活函数求导比较麻烦，并且在网络比较深的时候会出现梯度消失的情况


padding的大小是1和2, (1,2)：
    表示在特征矩阵的左边加上一列0，右边加上两列0，上面加上一列0，下面加上两列0
    (注意代表padding的2p值的是两边padding的像素之和，并不一定要求两边像素一定要一样)

padding 只能接收两种类型：
    1. int 整型
        e.g. p=1
        则会在图片的上边和下边补一行0，在左边和右边补一列0
    2. tuple 类型
        e.g. p=(1, 2)
        1代表上下方各补一行0,2代表左右两侧各补两列0
        # 若想精确的在左边补一列0，右边补两列0，则需使用 nn.ZeroPad2d() 方法。
        
        
在卷积和池化操作结果出现不为整数的情况：

    - https://blog.csdn.net/qq_37541097/article/details/102926037


https://www.bilibili.com/video/BV1p7411T7Pc


conda activate yolov5-6.2

20230905:
    watch -n 1 -c gpustat --color 
    



Reference:
https://blog.csdn.net/m0_37867091/article/details/107150142
https://www.bilibili.com/h5/note-app/view?cvid=14216422&pagefrom=comment
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

