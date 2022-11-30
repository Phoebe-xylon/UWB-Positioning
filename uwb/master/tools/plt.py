# -*- coding: utf-8 -*-
"""
__project_ = 'uwb'
__file_name__ = 'plt'
__author__ = 'Pianxy 1'
__time__ = '2021/10/16 21:07'
__product_name = PyCharm

# import necessary module
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load data from file
# you can replace this using with open
#data1 = pd.read_csv('../experiments/result.txt',sep=',').to_numpy()
data1 = pd.read_csv('./result.txt',sep=',',skiprows=0,header=None).to_numpy().T
'''
first_2000 = sorted(data1[:, 0])
second_2000 = sorted(data1[:, 1])
third_2000 = sorted(data1[:, 2])
'''
first_2000 = data1[::5, 0]
second_2000 = data1[::5, 1]
third_2000 = data1[::5, 2]

# print to check data
print (first_2000)
print (second_2000)
print (third_2000)

# new a figure and set it into 3d
fig = plt.figure()
ax = fig.gca(projection='3d')

# set figure information
ax.set_title("3D_Curve")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# draw the figure, the color is r = read
figure = ax.plot(first_2000, second_2000, third_2000, c='r')

plt.show()