# -*- utf-8 -*-
# @time:2021/1/2318:56,
# @auther: 骈鑫洋 2030601036
# @file:key frame
# @product：PyCharm

import os
import scipy.io as scio
import numpy as np
import pandas as pd
import random

#data_path ='D:/data/UWBdata/norm/'
#data_path ='D:/data/UWBdata/no-norm/'
data_path ='D:\data\data-UWB-nonormal/valu/'
filelist = os.listdir(data_path)
if __name__ == '__main__':
    for file in filelist:
        information=[]
        f = pd.read_csv(data_path+file, sep=':',skiprows=0,names=[1,2,3,4,5,6,7,8,9])
        f = pd.DataFrame(f[6].to_numpy().reshape((-1,4))).dropna()
        if 'nonorm' in data_path:
                np.save('D:\data\data-UWB-nonormal/valu/'+file[:-4],f)
                #scio.savemat('D:\data\data-UWB-zl/test/'+file[:-4]+'.mat',{"a": f})
        else:
                np.save('D:\data\data-UWB-normal/valu/'+file[:-4],f)
                #scio.savemat('D:\data\data-UWB-zl/test/'+file[:-4]+'.mat',{"a": f})


