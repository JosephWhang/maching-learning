# https://blog.csdn.net/kancy110/article/details/69665164
import numpy as np

# np.random.rand(r0,r1,r2...) 生成一个[0,1)之间的随机浮点数或N维浮点数组。
# 无参数
rdnp1 = np.random.rand()  # 默认生成[0,1]之间的随机数
print(rdnp1)
print(type(rdnp1))    # <class 'float'>

rdnp2 = np.random.rand(3) # 生成列表长度为3的列表
print(rdnp2)
print(type(rdnp2))  # <class 'numpy.ndarray'>

rdnp3 = np.random.rand(2,3) # 生成2*3的列表
print(rdnp3)
print(type(rdnp3))  # <class 'numpy.ndarray'>


# numpy.random.randn(d0, d1, ..., dn)：生成一个浮点数或N维浮点数组，取数范围：正态分布的随机样本数。
rdnnp1 = np.random.randn()  # 不一定是[0,1]的随机数之间的随机数
print(rdnnp1)
print(type(rdnnp1))    # <class 'float'>
# 其他情况同np.random.rand()


# numpy.random.standard_normal(size=None)：生产一个浮点数或N维浮点数组，取数范围：标准正态分布随机样本
stnp1 = np.random.standard_normal(2)
print(stnp1)
print(type(stnp1))

stnp2 = np.random.standard_normal((2,3))
print(stnp2)
print(type(stnp2))


# numpy.random.randint(low, high=None, size=None, dtype='l')：
# 生成一个整数或N维整数数组，取数范围：若high不为None时，取[low,high)之间随机整数，否则取值[0,low)之间随机整数。
rdinp1 = np.random.randint(10,15,size=4)
print(rdinp1)
print(type(rdinp1))


# numpy.random.random_integers(low, high=None, size=None)：
# 生成一个整数或一个N维整数数组，取值范围：若high不为None，则取[low,high]之间随机整数，否则取[1,low]之间随机整数。
rdit = np.random.random_integers(2,5,size=5)
print(rdit)
print(type(rdit))


# numpy.random.random_sample(size=None)：
# 生成一个[0,1)之间随机浮点数或N维浮点数组。
rs = np.random.random_sample((2, 3, 5))
print(rs)
print(type(rs))


# numpy.random.choice(a, size=None, replace=True, p=None)：
# 从序列中获取元素，若a为整数，元素取值为np.range(a)中随机数；若a为数组，取值为a数组元素中随机元素。
choi = np.random.choice(['a','b','c','d'])
print(choi)
print(type(choi))


# numpy.random.shuffle(x)：
# 对X进行重排序，如果X为多维数组，只沿第一条轴洗牌，输出为None。
list1 = [1,2,3,4,5]
shf = np.random.shuffle(list1)
print(list1)
print(shf)
print(type(shf))

list2 = np.arange(12).reshape(3,4)
shf2 = np.random.shuffle(list2)
print(list2)
print(shf2)
print(type(shf2))
# [[ 8  9 10 11]
#  [ 0  1  2  3]
#  [ 4  5  6  7]]


# numpy.random.permutation(x)：
# 与numpy.random.shuffle(x)函数功能相同，
# 两者区别：peumutation(x)不会修改X的顺序。
arr1 = [1,2,3,4,5]
pm = np.random.permutation(arr1)
print(arr1)
print(pm)
print(type(pm))