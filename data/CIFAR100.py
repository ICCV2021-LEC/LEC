import os
import random
import pickle
import numpy as np
from config import settings

def unpickle(file): 
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class NC_CIFAR100():
    def __init__(self, args, transform=None, c_way=5, k_shot=5):
        self.name = 'NC_CIFAR100'
        self.root = settings.CIFA100_DIR
        self.IndexDir = os.path.join(self.root, 'Index_list')
        self.Img, self.Label = self.Read_CIFAR(os.path.join(self.root, 'train'))    
        self.transform = transform
        self.count = 0
        self.Set_Session(args)
    
    def Set_Session(self, args):
        self.sess = args.sess
        self.Index_list = self.Read_Index_Sess()
        self.len = len(self.Index_list)
        print(len(self.Index_list))
    
    def Read_CIFAR(self, data_dir):
        a = unpickle(data_dir)
        X = a[b'data'].reshape(50000, 3, 32, 32).transpose(0,2,3,1)
        Y = a[b'fine_labels']
        
        return X, Y
    
    def Read_Index_Sess(self):
        idx = []
        f = open(self.IndexDir+'/session_'+str(self.sess+1)+'.txt', 'r')
        while True:
            lines = f.readline() 
            if not lines:
                break
            idx.append(int(lines.strip()))
            
        return idx
    
    def Random_choose(self):
        Index = np.random.choice(self.Index_list, 1, replace=False)[0]
        
        return Index
    
    def load_frame(self, Index):
        Image = self.Img[Index]
        Label = self.Label[Index]
        
        return Image, Label
    
    def __len__(self):
        return  self.len
    
    def __getitem__(self, idx):
        Index = self.Index_list[idx]
        Image, Label = self.load_frame(Index)
        if self.transform is not None:
            Image = self.transform(Image)
        self.count = self.count + 1

        return Image, Label
    
