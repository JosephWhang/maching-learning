from random import seed
from random import randrange
from csv import reader
import matplotlib.pyplot as plt
from math import sqrt
# dataset = [[1.2, 1.1], [2.4, 3.5], [4.1, 3.2], [3.4, 2.8], [5, 5.4]]    # 样本数据


# 导入CSV文件，文件内容为瑞典汽车保险数据库
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        headings = next(csv_reader)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# 将字符串列转换为浮点数
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# 将数据集分为训练集合和测试集合两部分
def train_test_split(dataset, percent):
    train = list()
    train_size = percent * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


# 定义一个求均值的函数
def mean(values):
    return sum(values) / float(len(values))


# 定义一个计算方差的函数
def variance(values, means):
    return sum([(x - means) ** 2 for x in values])


# 定义一个计算协方差的函数
def covariance(x, y, mean_x, mean_y):
    # 协方差
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


# 定义一个计算回归系数的函数
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    plt.axis([0, 150, 0, 450])
    plt.plot(x, y, 'bs')
    plt.grid()
    mean_x, mean_y = mean(x), mean(y)
    w1 = covariance(x, y, mean_x, mean_y) / variance(x, mean_x)
    w0 = mean_y - w1 * mean_x
    return w1, w0


# 计算均方差根误差 RMSE
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


# 计算预测值y
def simple_linear_regression(train, test):
    prediction = list()
    w1, w0 = coefficients(train)
    for row in test:
        y_model = w1 * row[0] + w0
        prediction.append(y_model)
    return prediction


# 评估算法数据准备及协调
def evaluate_algorithm(dataset, algorithm, split_percent, *args):
    train, test = train_test_split(dataset, split_percent)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    plt.plot([x[0] for x in test], predicted, 'r-')
    plt.show()

    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    return rmse


seed(2)
filename = 'test.csv'
dataset = load_csv(filename)
for col in range(len(dataset[0])):
    str_column_to_float(dataset, col)


# 设置数据集合分割百分比
percent = 0.6
rmse = evaluate_algorithm(dataset, simple_linear_regression, percent)

# 返回RMSE 均方根误差
print('RMSE: %.3f' % rmse)

'''
    # 分别计算x和y的均值和方差以及协方差
    mean_x, mean_y = mean(x), mean(y)
    var_x, var_y = variance(x, mean_x), variance(y, mean_y)
    covar = covariance(x, y, mean_x, mean_y)
    print('mean_x: %.3f, var_x: %.3f' % (mean_x, var_x))    # 输出平均数
    print('mean_y: %.3f, var_y: %.3f' % (mean_y, var_y))    # 输出方差
    print('covar: %.3f' % covar)    # 输出协方差
    # 作图，x：0~6和y：0~6
    plt.axis([0,6,0,6])
    plt.plot(x,y,'bs')
    plt.grid()
    plt.show()
'''

