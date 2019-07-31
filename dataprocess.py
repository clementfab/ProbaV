import torch
import os
from embiggen import *


DATA_PATH = '../data/probav_data/'

train_paths = all_scenes_paths(DATA_PATH + 'train/')
test_paths = all_scenes_paths(DATA_PATH + 'test/')


train_lr_median_all = [
    central_tendency(path, agg_with='median', only_clear=False)
    for path in tqdm(train_paths, desc='aggregate median')]

train_lr_mean_clear = [
    central_tendency(path, agg_with='mean', only_clear=True, fill_obscured=True)
    for path in tqdm(train_paths, desc='aggregate mean')]

test_lr_median_all = [
    central_tendency(path, agg_with='median', only_clear=False)
    for path in tqdm(test_paths, desc='aggregate median')]

test_lr_mean_clear = [
    central_tendency(path, agg_with='mean', only_clear=True, fill_obscured=True)
    for path in tqdm(test_paths, desc='aggregate mean')]


torch.save(train_lr_median_all, os.path.join(DATA_PATH, 'train', 'lr_median_all'))
torch.save(train_lr_mean_clear, os.path.join(DATA_PATH, 'train','lr_mean_clear'))
torch.save(test_lr_median_all, os.path.join(DATA_PATH, 'test', 'lr_median_all'))
torch.save(test_lr_mean_clear, os.path.join(DATA_PATH, 'test','lr_mean_clear'))