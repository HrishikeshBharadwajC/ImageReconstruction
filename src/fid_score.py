#!/usr/bin/env python
# coding: utf-8

import torch
from generator_model import Generator
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import *
from dataset import GenerateData
from utils import *
from scores import *

root_path = '../data/maps'
save_every_n_epochs = 10
final_epoch = 200
verbose=1


# In[12]:


val_dataset = GenerateData(root_path+'/val', direction_reverse=False)
batch_size= 64
num_workers=4
val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"


# In[14]:


fids = []
generator_ckp_list = [i*save_every_n_epochs + 1 for i in range(final_epoch//save_every_n_epochs-1)]
generator_ckp_list.append(final_epoch)
# generator_ckp_list = [1,500,251]
val_size = len(val_data_loader.sampler)
print(f"Val size = {val_size}")
for i in generator_ckp_list:
    fidscaled = 0.0 
    for idx, (x, Y_original) in enumerate(val_data_loader):
        checkpoint_path = f'{root_path}/checkpoints/{i}_generator_G.pth'
        generator = Generator().to(device)
        generator.eval()
        generator_checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(generator_checkpoint["state_dict"])
        Y_generated = generator(x.to(device))
        Y_generated = Y_generated.detach()
        Y_original = Y_original.to(device)
        Y_original = Y_original.detach()
        fidscaled += calculate_fid_score(Y_original, Y_generated)*x.size(0)
    fidscaled /= val_size
    if verbose ==1:
        print(f"Checkpoint Epoch {i} FID score {fidscaled}")
    fids.append(fidscaled)
    np.savetxt(f"{root_path}/results/fid_log.csv",fids,header="FID Score",delimiter=',')
    plot_fids(root_path, generator_ckp_list[:len(fids)],fids)
