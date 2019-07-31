from itertools import cycle

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from embiggen import *


class CustomDataset(Dataset):
    def __init__(self, root='../data/probav_data/', train=True):
        self.root = root
        self.train = train
        
        if train == True:
            self.train_paths = all_scenes_paths(root + 'train/')
            
            self.train_hr = [highres_image(scene_path)[0] for scene_path in self.train_paths]
            
            train_median_all = torch.load(os.path.join(self.root, 'train', 'lr_median_all'))
            train_mean_clear = torch.load(os.path.join(self.root, 'train', 'lr_mean_clear'))
            
            self.train_data = [[im1,im2] for im1,im2 in zip(train_median_all,train_mean_clear)]
            self.train_data = torch.DoubleTensor(self.train_data)
            
        else:
            self.test_paths = all_scenes_paths(root + 'test/')

            self.test_hr = [highres_image(scene_path)[0] for scene_path in self.test_paths]
            
            test_median_all = torch.load(os.path.join(self.root, 'test', 'lr_median_all'))
            test_mean_clear = torch.load(os.path.join(self.root, 'test','lr_mean_clear'))
            
            self.test_data = [[im1,im2] for im1,im2 in zip(test_median_all,test_mean_clear)]
            self.test_data = torch.DoubleTensor(self.test_data)
        
        
    def __len__(self):
        if self.train:
            return len(self.train_paths)
        else:
            return len(self.test_paths)
        
        
    def __getitem__(self, index):
        if self.train:
            imgs, hr, paths = self.train_data[index], self.train_hr[index], self.train_paths[index]
        else:
            imgs, hr, paths = self.test_data[index], self.test_hr[index], self.test_paths[index]
        
        return imgs, hr, paths
    
    
    
#Testing data loading
if __name__ == "__main__":
    dset = CustomDataset()
    loader = cycle(DataLoader(dset, batch_size=4, shuffle=True, num_workers=0, drop_last=True))
    batch_lr, batch_hr, batch_paths = next(loader)