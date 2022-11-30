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
        self.label_path = self.root[:-5]+'/label/Tag坐标信息.txt'
        self.label_dict = self.label_load(self.label_path)
        self.datapath = []
        file_list = os.listdir(self.root)
        for file in file_list:
            self.datapath.append(os.path.join(self.root,file))
        self.data = self.data_load(self.datapath,self.label_dict)
        #self.data = self.normalize(self.data)
    def __getitem__(self, index):
        data = self.data[index]
        point_set = data[:4]
        point_set = self.fix(point_set)
        point_set = torch.from_numpy(point_set)
        label = [data[4]/500,data[5]/500,data[6]/300]#500,500,250
        label = torch.tensor(label)
        return point_set, label

    def fix(self,point_set):
        data = []
        max = (300**2+500**2+500**2)**0.5
        one = pd.Series([0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        data.append(((pd.Series([500,300,300,0, 0, 120,0])+point_set[0]*one)/max).tolist())
        data.append(((pd.Series([500,300,300,500,0,160,0])+point_set[1]*one)/max).tolist())
        data.append(((pd.Series([500,300,300,0,300,160,0])+point_set[2]*one)/max).tolist())
        data.append(((pd.Series([500,300,300,500,300,120,0])+point_set[3]*one)/max).tolist())
        return np.array(data)
    def  label_load(self,label_path):
        f = pd.read_csv(label_path, sep=' ',skiprows=2, names=['label', 'x', 'y','z'])
        f.drop(['label'],axis=1,inplace=True)
        return f.to_numpy()

    def data_load(self,datapath,label):
        data = []
        for path in datapath:
            data_son = np.load(path,allow_pickle=True)
            if 'valu' not in path:
                label_index = int(path.split('\\')[-1].split('.')[0]) - 1
                label_son = label[label_index,:]
            else:
                label_son = [0,0,0]
            for single in data_son:
                data.append(np.hstack([single,label_son]))
        data = np.array(data)
        return data
    def __len__(self):
        return len(self.data)
       
    def normalize(self, pc):
        """ pc: NxC, return NxC """
        for col_idx in range(pc.shape[1]):
            col = pc[:,col_idx]
            pc[:,col_idx] = col/col.max()
        return pc


if __name__ == '__main__':
    dset = PartDataset( root='D:\data\data-UWB/train')
#    d = PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=False, class_choice=None, npoints=4096, split='test')
    for i,(input) in enumerate(dset):
        real,target = input
        print(real.shape)
        print(target)
    print(len(dset))
    ps, cls = dset[10]
    print(cls)

