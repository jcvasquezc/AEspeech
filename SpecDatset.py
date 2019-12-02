
import torch.utils.data as data
import os
import numpy as np

import torch

class SpecDataset(data.Dataset):
    def __init__(self, datafolder):
        self.datafolder = datafolder
        self.image_files_list =os.listdir(datafolder)


    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = self.datafolder+self.image_files_list[idx]

        mat=np.load(img_name)      

        return mat