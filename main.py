from brain_cancer import Model;
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset;
from bisect import bisect
import os, psutil
from torchvision.io import read_image
import nibabel



class ImageDataset(Dataset):
  def __init__(self, annotation_file, img_dir, transform=None, target_transform=None) -> None:
    super().__init__()
    self.img_lablels = pd.read_csv(annotation_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.img_lablels)

  def __getitem__(self, index):
    img_path = os.path.join(self.img_dir, "sub-" + self.img_lablels["participant_id", index] + "_preproc-cat12vbm_desc-gm_T1w.npy" )
    image = read_image(img_path)
    label = self.img_lablels.iloc["age", index]
    if self.transform :
      image = self.transform(image)
    if self.target_transform :
      label = self.target_transform(label)
    return image, label





class BigDataset(Dataset):
  def __init__(self, data_paths, target_paths) -> None:
    super().__init__()
    self.data_memmaps = [np.load(path, mmap_mode='r') for path in data_paths]
    self.target_memmaps = [np.load(path, mmap_mode='r') for path in target_paths]
    self.start_indices = [0] * len(data_paths)
    self.data_count = 0
    for index, memmap in enumerate(self.data_memmaps):
      self.start_indices[index] = self.data_count
      self.data_count += memmap.shape[0]
      
  def __len__(self):
    return self.data_count

  def __getitem__(self, index):
    memmap_index = bisect(self.start_indices, index) - 1
    index_in_memmap = index - self.start_indices[memmap_index]
    data = self.data_memmaps[memmap_index][index_in_memmap]
    target = self.target_memmaps[memmap_index][index_in_memmap]
    return index, torch.from_numpy(data.copy()), torch.from_numpy(target.copy())
  

if __name__ == "__main__" :
  # how get participant_id?
  df = pd.read_csv('./data/openBHB/train/participants.tsv', sep='\t')
  participant_id = df['participant_id']
  target_val = df['age']
  # print(['id is: '+ str(int(pid)) for pid in target_val])
  
  
  data_paths_CAT = [f'data/openBHB/train/sub-{index}_preproc-cat12vbm_desc-gm_T1w.npy' for index in participant_id]
  data_paths_FLS = [f'data/openBHB/train/sub-{index}_preproc-freesurfer_desc-xhemi_T1w.npy' for index in participant_id]
  data_paths_Quasi = [f'data/openBHB/train/sub-{index}_preproc-quasiraw_T1w.npy' for index in participant_id]
  target_paths = [f'data/openBHB/train/sub-{index}_preproc-cat12vbm_desc-gm_T1w.npy' for index in participant_id] # what is target?
  
  process = psutil.Process(os.getpid())
  memory_before = process.memory_info().rss
  #image = nibabel.load(data_paths_FLS[0]).get_fdata()
  #print(image.shape)
  
  dataset = BigDataset(data_paths_Quasi, target_paths)

  used_memory = process.memory_info().rss - memory_before
  print("Used memory:", used_memory, "bytes")

  dataset_size = len(dataset)
  print("Dataset size:", dataset_size)
  print("Samples:")
  #print(dataset.shape())
  print(dataset[0][1].shape)
  print(dataset[0][0])
  print(dataset[1][1].shape)
  print(dataset[1][0])
  print(dataset[100][1].shape)
  print(dataset[100][0])


  for sample_index in [0, dataset_size//2, dataset_size-1]:
    #print(dataset[sample_index])
    None
  
  train_model = Model()
  
  
  print("Done")
  exit()