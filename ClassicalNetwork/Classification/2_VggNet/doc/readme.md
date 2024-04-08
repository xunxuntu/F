
我觉得PPT中的感受野计算公式是 ： F(i) = (F(i-1) -1) X Stride + Ksize  其中 F(0)=0, i 表示从上往下数，第i层。

感受野的计算公式：
    https://www.bilibili.com/video/BV17p4y1B78h/?vd_source=baaf037cb6fd369d88a01fa458647798
    https://www.cnblogs.com/shine-lee/p/12069176.html


参数计算：
    卷积核的尺寸 * 卷积核的尺寸 * 输入特征矩阵的深度 * 卷积核的个数

3x3卷积核卷积之后得到的输出特征图的尺寸和输入特征图的尺寸是一样的，因为3x3卷积核的stride=1，padding=1，根据公式 out_size = (in_size - F_size + 2P )/S + 1 所以得到 out_size 等于 in_size


https://www.bilibili.com/video/BV1q7411T7Y6


conda activate yolov5-6.2

20230905:
    watch -n 1 -c gpustat --color