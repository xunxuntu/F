"""

参考链接：https://mozillazg.com/2016/12/python-super-is-not-as-simple-as-you-thought.html
date:2021.12.24
"""


# 先来个通俗易懂的
class AA:
    def add(self, x):
        y = x + 1
        print(y)


class BB(AA):
    def add(self, x):
        super().add(x)


bb = BB()
bb.add(2)  # 3


print('#' * 100)


# 单继承
class A:
    def __init__(self):
        self.n = 20

    def add(self, m):
        print('self is {0} @A.add'.format(self))
        self.n += m


class B(A):
    def __init__(self):
        self.n = 3

    def add(self, m):
        print('self is {0} @B.add'.format(self))
        super().add(m)
        self.n += 3


b = B()
b.add(2)
print(b.n)  # 不继承A中的n=20； m=2， n=3，2+3+3=8
# super().add(m) 调用父类方法 def add(self, m) 时,
# 此时父类中 self 并不是父类的实例而是子类的实例,
# 所以 super().add(m) 之后 self.n 的结果是 5 而不是 22


print('*' * 100)


# 多继承
class C(A):
    def __init__(self):
        self.n = 4

    def add(self, m):
        print('self is {0} @C.add'.format(self))
        super().add(m)
        self.n += 4


class D(B, C):
    def __init__(self):
        self.n = 5

    def add(self, m):
        print('self is {0} @D.add'.format(self))
        super().add(m)
        self.n += 5


d = D()
d.add(2)
print(d.n)
