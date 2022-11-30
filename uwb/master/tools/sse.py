import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#读取靶点坐标
def label_load(path):
    f = pd.read_csv(path.format('\label\Tag坐标信息.txt'), sep=' ', skiprows=2, names=['label', 'x', 'y', 'z'])
    f.drop(['label'], axis=1, inplace=True)
    return f.to_numpy()
#场景信息的应用
information_d = [0,0,120,500,0,160,0,300,160,500,300,120]
information_d = np.array(information_d).reshape([4,3])
information_d = np.linalg.norm(information_d, axis=1, keepdims=True).reshape([-1])
tar_information = np.array([500,300,300])
information = np.linalg.norm(tar_information, axis=0, keepdims=True).reshape([-1])

#文件位置
path = r'D:\data\data-UWB-normal\{}'
label = label_load(path)

#切分训练集和训练集 y =X[:,n], n=4的时候拟合x的表达式，n=5的时候拟合y的表达式，n =6的时候拟合z的表达式
n = 6

#对数据进行整合方便后面训练
def load_data(path,information,information_d,mode = 'train',):
    X = []
    file_list = os.listdir(path.format(mode))
    for file in file_list:
        data_son = np.load(path.format(mode+'/')+file, allow_pickle=True)
        if mode != 'valu':
            label_index = int(file.split('.')[0]) - 1
            label_son = label[label_index, :]
        else:
            label_son = [0, 0, 0]
        for single in data_son:
            single = (information+single)/information_d
            label_son = label_son
            X.append(np.hstack([single, label_son]))
    return np.array(X)
X = load_data(path,information,information_d,'train')

y =X[:,n].T
X_train,X_test,y_train,y_test = train_test_split(X[:,:4],y,random_state=666)

#进行线性拟合
reg = LinearRegression()
reg.fit(X_train,y_train)

#输出结果
print(reg.coef_)
print(reg.intercept_)
print(reg.score(X_test,y_test))


X = load_data(path,information,information_d,'valu')
result = np.dot(X[:,:4],reg.coef_.T)+reg.intercept_
f = open("./result.txt", 'a')
for num in result:
    f.write(str(num) + ',')
f.write('\n')
f.close()