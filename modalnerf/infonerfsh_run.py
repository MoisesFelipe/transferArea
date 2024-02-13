#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:44:35 2023

@author: felipe
"""
import numpy as np
import infonerf_impl_sh_blender_llff_func as infoNerfClass
# import multiprocessing as Process
import torch.multiprocessing as mp
from torch.multiprocessing import Process

class Process(Process):
    def __init__(self, id, model):
        
        super(Process, self).__init__()
        self.id = id
        
        self.model = model
        
    def run(self):
        self.model.train()
        self.model.render_test_imgs()
        

device = 'cuda:0'

sh_order = 3
sh_coeff_num = sum(np.linspace(0,sh_order,num=sh_order+1)*2 +1) # number of sh coefficients

nb_epochs=1
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


train_idx = np.linspace(0, 90, 10).astype(int) # plate


ext = "png"
exp ="plate"
exp_name = exp+"_"+str(len(train_idx))+"imgs_"+str(factor)+"factor_test_class"


batch_size = int(256*4)
lr=1e-4


#%%



device0 = 'cuda:0'
device1 = 'cuda:1'



ini_time_step = 0
end_time_step = 1

t_range = range(ini_time_step,end_time_step+1)

nerf_models = []
for t, count in zip(t_range, range(0,len(t_range))):
    exp_t =exp+str(t)
    exp_name = exp_t +"/"+ exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor"
    
    print(exp_name)
    
    if count==0:
        device=device0
    else:
        device=device1
    
    nerf_models.append( infoNerfClass.InfoNerfSH(exp_name, exp, ext, train_idx, 
                                                factor=factor, llff=llff, white_bkgd=white_bkgd, 
                                                half_res=half_res, spherify=spherify,num_val_img=num_val_img) )
    
    nerf_models[count].__run__(device, hidden_dim=256, sh_coeff_num=sh_coeff_num, 
                        batch_size = batch_size,lr=lr, nb_epochs=nb_epochs, 
                        nb_bins=nb_bins,T=T,lambda_mul=lambda_mul,
                        scheduler_freq=None)

#%%
for _ in nerf_models:
    print("rodando")
    _.train()
    
#%%


nerf_models = []
processes = []

mp.set_start_method('spawn')

for t, count in zip(t_range, range(0,len(t_range))):
    exp_t =exp+str(t)
    exp_name = exp_t +"/"+ exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor"
    
    print(exp_name)

    # nerf_models.append( infoNerfClass.InfoNerfSH(exp_name, exp, ext, train_idx, 
    #                                             factor=factor, llff=llff, white_bkgd=white_bkgd, 
    #                                             half_res=half_res, spherify=spherify,num_val_img=num_val_img) )
    
    # nerf_models[count].__run__(device, hidden_dim=256, sh_coeff_num=sh_coeff_num, 
    #                     batch_size = batch_size,lr=lr, nb_epochs=nb_epochs, 
    #                     nb_bins=nb_bins,T=T,lambda_mul=lambda_mul,
    #                     scheduler_freq=None)
    
    model = infoNerfClass.InfoNerfSH(exp_name, exp, ext, train_idx, 
                                    factor=factor, llff=llff, white_bkgd=white_bkgd, 
                                    half_res=half_res, spherify=spherify,num_val_img=num_val_img)
    
    model.__run__(device, hidden_dim=256, sh_coeff_num=sh_coeff_num, 
                  batch_size = batch_size,lr=lr, nb_epochs=nb_epochs, 
                  nb_bins=nb_bins,T=T,lambda_mul=lambda_mul, scheduler_freq=None)
    
    processes.append( Process(count, model) )
    processes[-1].start()
    
#%%




p = Process(0)
  
# Create a new process and invoke the
# Process.run() method
p.start()
  
# Process.join() to wait for task completion.
p.join()
p = Process(1)
p.start()
p.join()

#%%
processes = []
for model in nerf_models:
    processes.append(  multiprocessing.Process(target=model.train() )  )
    processes[-1].start()

# for model in nerf_models:
    # model.render_test_imgs(device)
        
    # nerf_models[count].train(device)
    # nerf_models[count].render_test_imgs(device)
    