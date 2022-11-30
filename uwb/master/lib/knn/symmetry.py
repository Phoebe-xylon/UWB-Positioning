# -*- coding: utf-8 -*-
"""
__project_ = 'Symmetry-Detection-of-Occluded-Point-Cloud-Using-Deep-Learning-master'
__file_name__ = 'symmetry'
__author__ = 'Pianxy 1'
__time__ = '2021/9/1 12:08'
__product_name = PyCharm
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃        ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import  torch
from sympy import symbols, solve, linsolve
import open3d as o3d
import  numpy as np

def fit_plane(point):
    A = np.zeros((3,3))
    B = np.zeros((3,1))
    for p in point:
        A[0, 0] += p[0]**2
        A[0,1] += p[0]*p[1]
        A[0,2] += p[2]
        A[1,1] += p[1]**2
        A[1,2] += p[2]
        A[2,2] += 1
        B[0,0] += p[0]*p[2]
        B[1,0] += p[1]*p[2]
        B[2,0] += p[2]
    x, y, z = symbols('x y z')
    f1 = A[0,0] * x + A[0,1]*y + A[0,2]*z - B[0,0]
    f2 = A[0,1] * x + A[1,1]*y + A[1,2]*z - B[1,0]
    f3 = A[0,2] * x + A[1,2]*y + A[2,2]*z - B[2,0]
    # 写法1
    para = solve([f1, f2, f3])

    # 写法2
    #print(linsolve([f1, f2, f3], (x, y, z)))
    return para[x], para[y], -1, para[z]

if __name__ == '__main__':
    pointx = torch.rand(1000,1)
    pointy= torch.rand(1000,1)
    pointz = pointx+pointy
    point = torch.cat((pointx,pointy,pointz),dim=1)

    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point)
    o3d.visualization.draw_geometries([point_cloud])

    A,B,C,D = fit_plane(point)
    print(A,B,C,D)