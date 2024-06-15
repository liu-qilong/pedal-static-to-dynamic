import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset

from src.tool.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Pedar_Dataset_static2dynamic(Dataset):
    def __init__(self, pedar_static: str, pedar_dynamic: str, sense_range: float = 600, dtype = torch.float32, transform = None, target_transform = None):
        self.pedar_static = pd.read_pickle(pedar_static)
        self.pedar_dynamic = pd.read_pickle(pedar_dynamic)
        self.index = self.pedar_static.index
        
        self.dtype = dtype
        self.sense_range = sense_range

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        # get corresponding static and dynamic pressure data
        # transform to torch tensor and divide by the sense range
        static_pressure = torch.from_numpy(self.pedar_static.loc[self.index[idx], :].values).type(self.dtype) / self.sense_range
        dynamic_pressure = torch.from_numpy(self.pedar_dynamic.loc[self.index[idx], :].values).type(self.dtype) / self.sense_range

        if self.transform:
            static_pressure = self.transform(static_pressure)
        if self.target_transform:
            dynamic_pressure = self.target_transform(dynamic_pressure)

        return static_pressure, dynamic_pressure
    
    def draw_heatmap(self, arr: np.array, l_mask_path: str = 'config/left_foot_mask.png', plot: bool = True, vmin: float = 0.0, vmax: float = 600.0):
        # load foot masks
        l_img = Image.open(l_mask_path)
        r_img = ImageOps.mirror(l_img)

        l_mask = np.array(l_img).astype(np.float64)
        r_mask = np.array(r_img).astype(np.float64)

        # detect pixels of area no.1~197 and store the corresponding indexes
        l_index = {}
        r_index = {}

        for n in range(0, 198):
            l_index[n] = np.where(l_mask == n + 1)
            r_index[n + 99] = np.where(r_mask == n + 1)

        # create left and right foot heatmap
        l_pedar = np.zeros(l_mask.shape)
        r_pedar = np.zeros(r_mask.shape)

        for idx, value in enumerate(arr):
            if idx <= 99:
                # filling left foot area
                l_pedar[l_index[idx]] = value * self.sense_range

            else:
                # filling right foot area
                r_pedar[r_index[idx]] = value * self.sense_range

        # plot heatmap
        if plot:
            fig, axs = plt.subplots(1, 2)
            
            im = axs[0].imshow(l_pedar, vmin=vmin, vmax=vmax)
            axs[0].set_title('left')
            axs[0].axis('off')
            fig.colorbar(im, ax=axs[0])

            im = axs[1].imshow(r_pedar, vmin=vmin, vmax=vmax)
            axs[1].set_title('right')
            axs[1].axis('off')
            fig.colorbar(im, ax=axs[1])

            plt.show()
