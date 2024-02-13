#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:45:03 2023

@author: felipe
"""

import random
import numpy as np
import infonerfsh as infoNerfClass     
import torch

device = 'cuda:0'

sh_order = 3
sh_coeff_num = sum(np.linspace(0,sh_order,num=sh_order+1)*2 +1) # number of sh coefficients

nb_epochs=140
hn=2
hf=6
nb_bins=192
T=0.1
lambda_mul=4
normalize_imgs = True
white_bkgd = True
half_res = True
spherify = False
llff = False
factor = 1
num_val_img = 2

n_train_imgs = 32
train_idx = np.linspace(0, n_train_imgs, 10).astype(int) # plate


ext = "png"
exp ="frame_"
datafolder = "plate"
exp_name = exp+"_"+str(len(train_idx))+"imgs_"+str(factor)+"factor_test_class"


batch_size = int(256*4)
lr=5e-5


#%% Running on serial

# n_timesteps = 126

# ini_time_step = 0
# end_time_step = int(n_timesteps/2)

ini_time_step = 40
end_time_step = int(126/2)
n_timesteps = end_time_step-ini_time_step


# t_range = range(ini_time_step,end_time_step,4 )
t_range = random.sample(range(ini_time_step,end_time_step), int(n_timesteps*0.3))
t_range.sort()

nerf_models = []

for t, count in zip(t_range, range(0,len(t_range))):
    exp_t =datafolder+"/"+exp+str(t).zfill(3)
    # exp_t =datafolder+"/"+exp+str(count).zfill(3)
    exp_name = exp_t +"/"+ exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device
    model_name = "model_"+exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device+".pth"
    
    print(exp_name)
    
    #%

    nerf_models.append( infoNerfClass.InfoNerfSH(exp_name, exp_t, ext, train_idx, 
                                                factor=factor, llff=llff, white_bkgd=white_bkgd, 
                                                half_res=half_res, spherify=spherify,num_val_img=num_val_img) )
    
    nerf_models[count].__run__(device, hidden_dim=256, sh_coeff_num=sh_coeff_num, 
                        batch_size = batch_size,lr=lr, nb_epochs=nb_epochs, 
                        nb_bins=nb_bins,T=T,lambda_mul=lambda_mul,
                        scheduler_freq=None)
    
    nerf_models[count].train()
    
    nerf_models[count].render_test_imgs()
    
    # torch.save(nerf_models[count].model.state_dict(), "./"+exp_name+"/model_"+device+".pth")
    torch.save(nerf_models[count].model.state_dict(), "./"+exp_name+"/"+model_name+".pth")
    
    
    