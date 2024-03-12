

inverted_residual_setting = [
            # t, c, n, s
            # t: 扩展因子，例如将输出特征矩阵的深度调整为tk
            # c：输出的channel
            # n: bottleneck(倒残差结构)重复的次数
            # s: 每个bottleneck的第一层的卷积的步距s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

for t, c, n, s in inverted_residual_setting:
    print(f't: {t}')
    print(f'c: {c}')
    print(f'n: {n}')
    print(f's: {s}')
    print(f'=====================')