import numpy as np
import h5py
import json
import torch
import torch.utils.data as data
import os
import pickle
from multiprocessing import Pool

class SuppressDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):
        self.subset = subset
        self.mode = opt["mode"]
        self.data_file = h5py.File(opt["suppress_label_file"].format(self.subset), 'r')
        self.video_list = list(self.data_file.keys())
        self.inputs=[]
        for index in range(0,len(self.video_list)):
            video_name=self.video_list[index]
            duration = self.data_file[video_name+'/input'].shape[0]
            for i in range(0, duration):
                self.inputs.append([video_name,i])
                
        print ("%s subset seg numbers: %d" %(self.subset,len(self.inputs)))
        
    def __getitem__(self, index):
        video_name, idx = self.inputs[index]     
        
        input_seq = self.data_file[video_name+'/input'][idx]
        label = self.data_file[video_name+'/label'][idx]
        
        input_seq= torch.from_numpy(input_seq)
        label = torch.from_numpy(label)
        
        return input_seq, label
            
    def __len__(self):
        return len(self.inputs)
            
