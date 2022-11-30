#from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import pandas as pd
import os
import numpy as np


class PartDataset(data.Dataset):
    def __init__(self, root='D:\python_project\drew-point-cloud\data-1/train', phase = 'train',normalize=True):
        self.root = root
        self.phase = phase
        self.datapath = []
        file_list = os.listdir(self.root)
        for file in file_list:
            self.datapath.append(os.path.join(self.root,file))
        self.data = self.data_load(self.datapath)
        #self.data = self.normalize(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        input = []
        for data1 in data[:4]:
            for data2 in data[:4]:
                input.append(data1-data2)
        point_set = np.array(input)
        point_set = self.normalize(point_set)
        point_set = torch.from_numpy(point_set)
        label = data[4]
        label = torch.tensor(label)
        return point_set, label


    def data_load(self,datapath):
        data = []
        for path in datapath:
            data_son = np.load(path,allow_pickle=True)
            if 'valu' not in path:
                if path.split('/')[-1].split('.')[1] == '正常':
                    label_son = 0
                else:
                    label_son = 1
            else:
                label_son = 0
            for single in data_son:
                data.append(np.hstack([single,label_son]))
        data = np.array(data)
        return data




    def __len__(self):
        return len(self.data)
       
    def normalize(self, pc):
        """ pc: NxC, return NxC """
        pc = (pc -pc.mean())/(pc.max()-pc.min())
        return pc


if __name__ == '__main__':
    dset = PartDataset( root='D:\data\data-UWB-data/train')
#    d = PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=False, class_choice=None, npoints=4096, split='test')
    for i,(input) in enumerate(dset):
        real,target = input
        print(real.shape)
        print(target)
    print(len(dset))
    ps, cls = dset[10]
    print(cls)
#    print(ps.size(), ps.type(), cls.size(), cls.type())
#    print(ps)
#    ps = ps.numpy()
#    np.savetxt('ps'+'.txt', ps, fmt = "%f %f %f")
