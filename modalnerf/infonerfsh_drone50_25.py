#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:58:58 2023

@author: felipe
"""


import random
import numpy as np
import infonerfsh as infoNerfClass     
import torch

device = 'cpu'

sh_order = 3
sh_coeff_num = sum(np.linspace(0,sh_order,num=sh_order+1)*2 +1) # number of sh coefficients

nb_epochs=5
hn=2
hf=6
nb_bins=192
T=0.1
lambda_mul=4
normalize_imgs = True
white_bkgd = False
half_res = True
spherify = False
llff = False
factor = 1
num_val_img = 2

n_train_imgs = 32
train_idx = np.linspace(0, n_train_imgs, 31).astype(int) 


ext = "png"
exp =""
datafolder = "drone"
exp_name = exp+"_"+str(len(train_idx))+"imgs_"+str(factor)+"factor_test_class"


batch_size = int(256*20)
lr=5e-5


#%% Running

exp_t =datafolder+"/"+exp
# exp_t =datafolder+"/"+exp+str(count).zfill(3)
exp_name = exp_t +"/"+ exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device
model_name = "model_"+exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device+".pth"

print(exp_name)

#%

nerf_model =  infoNerfClass.InfoNerfSH(exp_name, exp_t, ext, train_idx, 
                                            factor=factor, llff=llff, white_bkgd=white_bkgd, 
                                            half_res=half_res, spherify=spherify,num_val_img=num_val_img)

nerf_model.__run__(device, hidden_dim=256, sh_coeff_num=sh_coeff_num, 
                    batch_size = batch_size,lr=lr, nb_epochs=nb_epochs, 
                    nb_bins=nb_bins,T=T,lambda_mul=lambda_mul, 
                    scheduler_freq=None)

nerf_model.train()

nerf_model.render_test_imgs()

#%%

# nerf_model.model_optimizer = torch.optim.Adam(nerf_model.model.parameters(), lr=5e-6)
nerf_model.lr = 5e-5
nerf_model.train()

nerf_model.render_test_imgs()


# torch.save(nerf_model.model.state_dict(), "./"+exp_name+"/"+model_name+".pth")

# nerf_model.model_optimizer = torch.optim.Adam(nerf_model.model.parameters(), lr=1e-5)

#%%

for i in range(0,nerf_model.n_train_imgs):
    nerf_model.train_plot(nerf_model.model, nerf_model.hn, nerf_model.hf, nerf_model.training_dataset, 
             nerf_model.device, nerf_model.training_path, chunk_size=10, 
             img_index=i, nb_bins=nerf_model.nb_bins, 
             H=nerf_model.H, W=nerf_model.W)
    

#%%

for i in range(0,nerf_model.n_train_imgs):
    nerf_model.val(nerf_model.model, nerf_model.hn, nerf_model.hf, nerf_model.training_dataset, 
         nerf_model.device, nerf_model.epoch, nerf_model.training_path, chunk_size=10, 
         img_index=i, nb_bins=nerf_model.nb_bins, 
         H=nerf_model.H, W=nerf_model.W, normalize=nerf_model.normalize_imgs)

