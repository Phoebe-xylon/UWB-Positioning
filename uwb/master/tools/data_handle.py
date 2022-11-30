# -*- utf-8 -*-
# @time:2021/1/2318:56,
# @auther: 骈鑫洋 2030601036
# @file:key frame
# @product：PyCharm

import os
import numpy as np
import pandas as pd
import random

# data_path ='D:/data/UWBdata/norm/'
data_path = 'D:/data/UWBdata/no-norm/'
filelist = os.listdir(data_path)
num = 0
if __name__ == '__main__':
    testlist = random.sample(filelist, 65)
    test = []
    for file in filelist:
        information = []
        f = pd.read_csv(data_path + file, sep=':', skiprows=1, names=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        f = pd.DataFrame(f[6].to_numpy().reshape((-1, 4))).dropna()
        # print(f.describe())
        if 'no-norm' not in data_path:
            for line in f.columns:
                mean = f[line].mean()
                std = f[line].std()
                top = mean + 3 * std
                low = mean - 3 * std
                top_index = f.loc[f[line] > top].index
                if len(top_index) != 0:
                    f = f.drop(top_index)
                low_index = f.loc[f[line] < low].index
                if len(low_index) != 0:
                    f = f.drop(low_index)
        else:
            for line in f.columns:
                mean = f[line].mean()
                std = f[line].std()
                '''
                if std>50:
                    #
                    if f.loc[f[range(round(f.axes[0]/2)),line]].std>f.loc[f[range(round(f.axes[0]/2),f.axes[0]),line]].std:
                        no_norm=f.loc[f[range(round(f.axes[0] / 2), f.axes[0]), line]].index
                        nor_mean = f.loc[no_norm, line].mean()
                        nor_std = f.loc[no_norm, line].std()
                        no_norm=f.loc[f[range(round(f.axes[0]/2)),line]].index
                        f.loc[no_norm, line] = nor_mean+random.uniform(-nor_std, +nor_std )
                    else:
                        no_norm=f.loc[f[range(round(f.axes[0] / 2)), line]].index
                        nor_mean = f.loc[no_norm, line].mean()
                        nor_std = f.loc[no_norm, line].std()
                        no_norm=f.loc[f[range(round(f.axes[0] / 2), f.axes[0]), line]].index
                        f.loc[no_norm, line] = nor_mean+random.uniform(-nor_std, +nor_std )
                    #
                    for fre in range(frequence):
                        nor_index = f.loc[f[line] <mean].index
                        nor_mean = f.loc[nor_index, line].mean()
                        nor_std = f.loc[nor_index, line].std()
                        nor_index = f.loc[f[line] > mean].index
                        f.loc[nor_index, line] = nor_mean+random.uniform(-nor_std,+nor_std )
                    print(nor_std)
                mean = f[line].mean()
                std = f[line].std()
                '''
                top = mean + 3 * std
                low = mean - 3 * std
                top_index = f.loc[f[line] > top].index
                if len(top_index) != 0:
                    f = f.drop(top_index)
                low_index = f.loc[f[line] < low].index
                if len(low_index) != 0:
                    f.drop(low_index)
        f = f.drop_duplicates().to_numpy()

        num += f.shape[0]

        t = open('./process_norm.txt', 'a', encoding='utf-8')
        t.write(file + ':' + str(f.shape[0]) + '\n')
        t.close()

        if 'no-norm' in data_path:
            if file in testlist:
                np.save('D:\data\data-UWB-nonormal/test/' + file[:-4], f)
                # scio.savemat('D:\data\data-UWB-zl/test/'+file[:-4]+'.mat',{"a": f})
            else:
                np.save('D:\data\data-UWB-nonormal/train/' + file[:-4], f)
                # scio.savemat('D:\data\data-UWB-zl/train/'+file[:-4]+'.mat',{"a": f})
        else:
            if file in testlist:
                np.save('D:\data\data-UWB-normal/test/' + file[:-4], f)
                # scio.savemat('D:\data\data-UWB-zl/test/'+file[:-4]+'.mat',{"a": f})
            else:
                np.save('D:\data\data-UWB-normal/train/' + file[:-4], f)
                # scio.savemat('D:\data\data-UWB-zl/train/'+file[:-4]+'.mat',{"a": f})

    print(num)
