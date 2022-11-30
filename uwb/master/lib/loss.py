from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import open3d as o3d
from sympy import symbols, solve, linsolve
from tools.utils import PointLoss ,Related,sym_loss
import time
import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# from lib.knn.__init__ import KNearestNeighbor

def fit_plane(point):
    A = torch.zeros(3,3)
    B = torch.zeros(3,1)
    x = point[:,0]
    y = point[:,1]
    z = point[:,2]
    A[0,0] = torch.norm(x,2)**2
    A[0,1] = torch.dot(x.T,y)
    A[0,2] = torch.norm(z,1)
    A[1,1] = torch.norm(y,2)**2
    A[1,2] = torch.norm(y,1)
    A[2,2] = torch.tensor([len(point)])
    B[0,0] = torch.dot(x.T,z)
    B[1,0] = torch.dot(y.T,z)
    B[2,0] = torch.norm(z,1)
    x, y, z = symbols('x y z')
    f1 = A[0,0] * x + A[0,1]*y + A[0,2]*z - B[0,0]
    f2 = A[0,1] * x + A[1,1]*y + A[1,2]*z - B[1,0]
    f3 = A[0,2] * x + A[1,2]*y + A[2,2]*z - B[2,0]
    # 写法1
    para = solve([f1, f2, f3])

    # 写法2
    #print(linsolve([f1, f2, f3], (x, y, z)))
    return -para[x], -para[y], -1, -para[z]


    '''
    #test = np.array(np.vstack((half1, half2)))
    point_half1 = o3d.PointCloud()
    point_half1.points = o3d.utility.Vector3dVector(np.array(half1_no))
    point_half1.paint_uniform_color([0, 0, 1.0])
    point_half2 = o3d.PointCloud()
    print(len(half2))
    point_half2.points = o3d.utility.Vector3dVector(np.array(half2))
    point_half2.paint_uniform_color([1.0, 0, 0])
    o3d.visualization.draw_geometries([point_half1,point_half2])
    '''

    half1 = torch.from_numpy(np.array(half1).astype(float)).unsqueeze(0)
    half2 = torch.from_numpy(np.array(half2).astype(float)).unsqueeze(0)
    return half1 , half2, half1_dis ,half2_dis

def loss_calculation(pred_norm, pred_on_plane ,idx, points):
    num_p,_= pred_on_plane.size()
    '''
    if pred_on_plane.max()<0.85:
        idex  = pred_on_plane.view(num_p,-1) > pred_on_plane.mean()
    else:
        idex = pred_on_plane.view(num_p, -1) > 0.85
    point = torch.masked_select(points[1], idex).view(-1,3)
    A, B, C, D=fit_plane(point)
    half1,half2 = symmtery(A, B, C, D,points[0])
    '''
    pred_norm = pred_norm.squeeze(0)
    #half1,half2, half1_dis ,half2_dis = symmtery(pred_norm[0],pred_norm[1],pred_norm[2],pred_norm[3],points[0])

    mse_loss = torch.nn.MSELoss(reduce=True, size_average=False)
    chamfer = PointLoss()
    Reloss = Related()
    #loss = torch.log(5*chamfer(half1,half2)+1.3*np.exp(half1.shape[1]-500))
    loss = torch.log(5*chamfer(pred_norm[0],pred_norm[1],pred_norm[2],pred_norm[3],points[0])+1.5*pred_norm[2]/(pred_norm[0]+pred_norm[1]))
    #loss = torch.log(5*chamfer(half1,half2)+1.5*pred_norm[2]/(pred_norm[0]+pred_norm[1])+1.3*np.exp(half1.shape[1]-500))
    #loss = [5*chamfer(half1,half2)+1.5*pred_norm[2]/(pred_norm[0]+pred_norm[1])+1.3*np.exp(half1.shape[1]-500)]
    #loss = [np.abs(Reloss(half1,half2))+1.3*np.exp(half1.shape[1]-500)]
    #loss = mse_loss(half1, half2)
    #loss = dist2 = F.pairwise_distance(half1, half2, p=2)
    #if int(loss[0]) <= 4.1:
    # print(loss)
    return loss


class Loss(_Loss):

    def __init__(self):
        super(Loss, self).__init__(True)

    def forward(self, pred_norm, pred_on_plane, idx, points):

        return loss_calculation(pred_norm, pred_on_plane,  idx, points,)


