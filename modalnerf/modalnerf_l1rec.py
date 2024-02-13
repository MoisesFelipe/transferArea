#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:45:29 2023

@author: felipe
"""

import random
import numpy as np
import infonerfsh as infoNerfClass     
import torch
import os

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def render_rays(nerf_model, sh_coeffs, sigma,  ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192, T=0.1):
    
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # print(ray_origins.unsqueeze(1).shape)
    
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%
    
    # sh_coeffs, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%
    
    # evaluating SH function
    sh_coeffs = sh_coeffs.swapaxes(1, 2)
    print("sh_coeffs: "+str(sh_coeffs.reshape(
        *sh_coeffs.shape[:-1],
        -1,
        (nerf_model.sh_order + 1) ** 2).shape))
    
    
    print("ray_directions: "+str(ray_directions.reshape(-1, 3).shape))
    
    colors = nerf_model.eval_sh_coeff(nerf_model.sh_order, sh_coeffs.reshape(
        *sh_coeffs.shape[:-1],
        -1,
        (nerf_model.sh_order + 1) ** 2), ray_directions.reshape(-1, 3))
        
    # print("color: "+str(colors.shape))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])
    # print("colors_reshape: "+str(colors.shape))
    
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = nerf_model.compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    prob = alpha / (alpha.sum(1).unsqueeze(1) + 1e-10)
    mask = alpha.sum(1).unsqueeze(1) > T
    regularization = -1 * prob * torch.log2(prob + 1e-10)
    regularization = (regularization * mask).sum(1).mean()
    
    c = (weights * colors).sum(dim=1)  # Pixel values
    # Regularization for white background
    weight_sum = weights.sum(-1).sum(-1)
    
    eps = 1e-10
    acc = weight_sum
    inv_eps = 1 / eps
    
    # print("weights: "+str(weights.shape))
    # print("t: "+str(t.shape))
    # depth = (weights * t).sum(axis=-1)
    depth = (weights * t[:,:,None]).sum(axis=-1).sum(-1)
    # print("depth: "+str(depth.shape))
    # print("acc: "+str(acc.shape))
    disp = acc / depth
    disp = torch.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
    
    # print("c: "+str(c.shape))
    # print("weights: "+str(weight_sum.unsqueeze(-1).shape))
    return c + 1 - weight_sum.unsqueeze(-1), regularization, disp[:,None], acc[:,None], weights, sigma, x,colors,sh_coeffs

@torch.no_grad()
def pc_extraction(nerf_obj, hn, hf, dataset, epoch, device, path_imgs, chunk_size=512, img_index=0, nb_bins=192, H=400, W=400,normalize=False):
    
    model = nerf_obj.model
    model.eval()
    
    # Check whether the specified path exists or not
    if path_imgs is not None:
        if not os.path.exists(path_imgs): # create it
            # Create a new directory because it does not exist
            os.makedirs(path_imgs)
            print("Image directory has been created!")
        
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    img_orig = dataset[img_index * H * W: (img_index + 1) * H * W, 6:9].data.cpu().numpy().reshape(H, W, 3)
    
    datapx,datadisp,dataacc = [],[],[]
    datasigma = np.empty((0,nb_bins))
    dataweights = np.empty((0,nb_bins))
    
    datapxcoords = np.empty((0,nb_bins,3))
    datacolors = np.empty((0,nb_bins,3))
    
    datacoeffs = np.empty((0,3,int(sh_coeff_num)))
    
    plt.ioff()
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        
        # render_rays(self, nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192, T=0.1):
        regenerated_px_values,regularization,disp,acc,weights,sigma, coords,colors,sh_coeffs = nerf_obj.render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        
        # print("weights: "+str(weights[:,:,0].shape))
        # print("sigma: "+str(sigma.shape))
        # print("coords: "+str(coords.cpu().numpy().shape))
        # print("sh_coeffs: "+str(sh_coeffs.cpu().numpy().shape))
        
        datapx.append(regenerated_px_values)
        datadisp.append(disp.repeat(1,3) )
        dataacc.append(acc.repeat(1,3))
        
        datasigma = np.concatenate([datasigma, sigma.cpu().numpy()], 0)
        dataweights = np.concatenate([dataweights, weights[:,:,0].cpu().numpy()], 0)
        
        datapxcoords = np.concatenate([datapxcoords, coords.cpu().numpy()], 0)
        datacolors = np.concatenate([datacolors, colors.cpu().numpy()], 0)
        
        datacoeffs = np.concatenate([datacoeffs, sh_coeffs.cpu().numpy()], 0)
            
        
    if path_imgs is not None:
        # saving rgb
        img = torch.cat(datapx).data.cpu().numpy().reshape(H, W, 3)
        
        # img = to8b(torch.cat(datapx).data.cpu().numpy()).reshape(H, W, 3)
        if normalize:
            # img *= 255
            img = (255*(img - np.min(img))/np.ptp(img)).astype(int)   
            img_orig = (255*(img_orig - np.min(img_orig))/np.ptp(img_orig)).astype(int)   
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img_orig)
        plt.title("test-set")
        plt.subplot(1,2,2)
        plt.imshow(img)
        plt.title("rendered")
        # plt.imshow(img)
        plt.savefig(f'{path_imgs}/img_{epoch}_val_{img_index}.png', bbox_inches='tight')
        plt.close()
        
        # saving disparity
        disp_img = torch.cat(datadisp).data.cpu().numpy().reshape(H, W, 3)
        plt.figure()
        # plt.imshow(disp_img.astype(int))
        plt.imshow(disp_img)
        plt.savefig(f'{path_imgs}/disp_{epoch}_val_{img_index}.png', bbox_inches='tight')
        plt.close()
        
        # saving depth
        depth_img = torch.cat(dataacc).data.cpu().numpy().reshape(H, W, 3)
        plt.figure()
        # plt.imshow(depth_img.astype(int))
        plt.imshow(depth_img)
        plt.savefig(f'{path_imgs}/depth_{epoch}_val_{img_index}.png', bbox_inches='tight')
        plt.close()    
    
    
    return datasigma,dataweights,datapxcoords,datapx,datadisp,dataacc,datacolors,datacoeffs



@torch.no_grad()
def pc_extraction_rec(nerf_obj, sh_coeffs, sigma, hn, hf, dataset, epoch, device, path_imgs, chunk_size=512, img_index=0, nb_bins=192, H=400, W=400,normalize=False):
    
    model = nerf_obj.model
    model.eval()
    
    # Check whether the specified path exists or not
    if path_imgs is not None:
        if not os.path.exists(path_imgs): # create it
            # Create a new directory because it does not exist
            os.makedirs(path_imgs)
            print("Image directory has been created!")
        
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    img_orig = dataset[img_index * H * W: (img_index + 1) * H * W, 6:9].data.cpu().numpy().reshape(H, W, 3)
    
    datapx,datadisp,dataacc = [],[],[]
    datasigma = np.empty((0,nb_bins))
    dataweights = np.empty((0,nb_bins))
    
    datapxcoords = np.empty((0,nb_bins,3))
    datacolors = np.empty((0,nb_bins,3))
    
    datacoeffs = np.empty((0,3,int(sh_coeff_num)))
    
    plt.ioff()
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        
        # render_rays(self, nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192, T=0.1):
        regenerated_px_values,regularization,disp,acc,weights,_, coords,colors,_ = render_rays(nerf_obj, 
                                                                                               sh_coeffs, sigma,
                                                                                               ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        
        # print("weights: "+str(weights[:,:,0].shape))
        # print("sigma: "+str(sigma.shape))
        # print("coords: "+str(coords.cpu().numpy().shape))
        # print("sh_coeffs: "+str(sh_coeffs.cpu().numpy().shape))
        
        datapx.append(regenerated_px_values)
        datadisp.append(disp.repeat(1,3) )
        dataacc.append(acc.repeat(1,3))
        
        datasigma = np.concatenate([datasigma, sigma.cpu().numpy()], 0)
        dataweights = np.concatenate([dataweights, weights[:,:,0].cpu().numpy()], 0)
        
        datapxcoords = np.concatenate([datapxcoords, coords.cpu().numpy()], 0)
        datacolors = np.concatenate([datacolors, colors.cpu().numpy()], 0)
        
        datacoeffs = np.concatenate([datacoeffs, sh_coeffs.cpu().numpy()], 0)
            
        
    if path_imgs is not None:
        # saving rgb
        img = torch.cat(datapx).data.cpu().numpy().reshape(H, W, 3)
        
        # img = to8b(torch.cat(datapx).data.cpu().numpy()).reshape(H, W, 3)
        if normalize:
            # img *= 255
            img = (255*(img - np.min(img))/np.ptp(img)).astype(int)   
            img_orig = (255*(img_orig - np.min(img_orig))/np.ptp(img_orig)).astype(int)   
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img_orig)
        plt.title("test-set")
        plt.subplot(1,2,2)
        plt.imshow(img)
        plt.title("rendered")
        # plt.imshow(img)
        plt.savefig(f'{path_imgs}/img_{epoch}_val_{img_index}.png', bbox_inches='tight')
        plt.close()
        
        # saving disparity
        disp_img = torch.cat(datadisp).data.cpu().numpy().reshape(H, W, 3)
        plt.figure()
        # plt.imshow(disp_img.astype(int))
        plt.imshow(disp_img)
        plt.savefig(f'{path_imgs}/disp_{epoch}_val_{img_index}.png', bbox_inches='tight')
        plt.close()
        
        # saving depth
        depth_img = torch.cat(dataacc).data.cpu().numpy().reshape(H, W, 3)
        plt.figure()
        # plt.imshow(depth_img.astype(int))
        plt.imshow(depth_img)
        plt.savefig(f'{path_imgs}/depth_{epoch}_val_{img_index}.png', bbox_inches='tight')
        plt.close()    
    
    
    return datasigma,dataweights,datapxcoords,datapx,datadisp,dataacc,datacolors,datacoeffs

#%%
device = 'cuda:0'
device0 = 'cuda:0'
device1 = 'cuda:1'

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

n_timesteps = 126

ini_time_step = 0
end_time_step = n_timesteps

# t_range = range(ini_time_step,end_time_step,4 )
# t_range = random.sample(range(ini_time_step,end_time_step), int(n_timesteps*0.3))
t_range = random.sample(range(0,end_time_step), int(n_timesteps))
t_range.sort()

nerf_models = []

#%%

# skip_time_steps = [6, 7, 17, 22, 26,31,32,65,86,89]
skip_time_steps = []
img_index = 0
th = 1.4

model_counter = 0

for t, count in zip(t_range, range(0,len(t_range))):
    
    if t is skip_time_steps:
        continue
    
    exp_t =datafolder+"/"+exp+str(t).zfill(3)
    
    exp_name0 = exp_t +"/"+ exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device0
    model_name0 = "model_"+exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device0+".pth"
    
    exp_name1 = exp_t +"/"+ exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device1
    model_name1 = "model_"+exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device1+".pth"
    
    
    model_path0 = "./"+exp_name0+"/"+model_name0+".pth"
    model_path1 = "./"+exp_name1+"/"+model_name1+".pth"
    
    
    if os.path.isfile(model_path0):
        model_path = model_path0
        exp_name = exp_name0
        model_name = model_name0
    elif os.path.isfile(model_path1):
        model_path = model_path1
        exp_name = exp_name1
        model_name = model_name1
    else:
        continue
        
    print(exp_name)
    
    
    print("loading model...")
    
    nerf_obj = infoNerfClass.InfoNerfSH(exp_name, exp_t, ext, train_idx, 
                                                factor=factor, llff=llff, white_bkgd=white_bkgd, device=device,
                                                half_res=half_res, spherify=spherify,num_val_img=num_val_img) 
    nerf_obj.__run__(device, hidden_dim=256, sh_coeff_num=sh_coeff_num, 
                        batch_size = batch_size,lr=lr, nb_epochs=nb_epochs, 
                        nb_bins=nb_bins,T=T,lambda_mul=lambda_mul,
                        scheduler_freq=None)
    
    nerf_obj.model.load_state_dict(torch.load(model_path))
    
    nerf_models.append( nerf_obj )
    
    #%
    
    sigma_path = "./"+datafolder+"/sigma/"
    
    datasigma,dataweights,datacoords,datapx,datadisp,dataacc,datacolors,datacoeffs = pc_extraction(nerf_obj, nerf_obj.hn, nerf_obj.hf, nerf_obj.full_dataset, 
                                                                                                0, device, None,chunk_size=4, img_index=img_index, nb_bins=nb_bins, 
                                                                                                H=nerf_obj.H, W=nerf_obj.W,normalize=normalize_imgs)
    
    # keep same coordinates
    if model_counter==0:
        datasigma_filt = datasigma.max(axis=1)
        non_neg_px = datasigma_filt>th
        datasigma_filt = datasigma_filt[non_neg_px]
        
        datasigma_filt_idx = datasigma[non_neg_px].argmax(axis=1)
        
        coeff_dataset = np.empty((0, int(sum(non_neg_px)*3*sh_coeff_num)) )
        sigma_dataset = np.empty((0, int(sum(non_neg_px) ) ))
    
    
    
    datasigma_positive = datasigma[non_neg_px]
    datasigma_positive = datasigma_positive.max(axis=1)[...,None].transpose()
    
    datacoeffs2 = datacoeffs.reshape(int(len(datacoeffs)/nb_bins), nb_bins, 3,int(sh_coeff_num))
    datacoeffs2 = datacoeffs2[non_neg_px,datasigma_filt_idx,:,:]
    datacoeffs2 = datacoeffs2.reshape(datacoeffs2.shape[0],datacoeffs2.shape[1]*datacoeffs2.shape[2] )
    datacoeffs2 = datacoeffs2.flatten()
    datacoeffs2 = datacoeffs2[...,None].transpose()
    
    # datacoeffs2.reshape(sum(non_neg_px),  3,int(sh_coeff_num)) # back to original shape
    coeff_dataset = np.concatenate([coeff_dataset, datacoeffs2])
    sigma_dataset = np.concatenate([sigma_dataset, datasigma_positive])
    
    datacoords_positive = datacoords[non_neg_px,datasigma_filt_idx,:]
    datacolors_positive = datacolors[non_neg_px,datasigma_filt_idx,:]

    datacolors_positive = (1*(datacolors_positive - np.min(datacolors_positive))/np.ptp(datacolors_positive))
    
    datacoords_positive_norm = (1*(datacoords_positive - np.min(datacoords_positive))/np.ptp(datacoords_positive))

    print("Number of 3D Locations: "+str(datacoords_positive_norm.shape))
    
    #%


    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    sc = ax.scatter3D(datacoords_positive_norm[:,0], datacoords_positive_norm[:,1], datacoords_positive_norm[:,2], c = datacolors_positive)
    ax.set_xlim([0.,1])
    ax.set_ylim([0.,1])
    ax.set_zlim([0.,1])
    plt.title("test-img "+str(img_index)+" | Sigma Th.: "+str(th)+" | Timestep: "+str(t)+" | Number of 3D Points: "+str(len(datacoords_positive_norm)) )
    ax.set_xlabel('X',fontsize=16,linespacing=50.1)
    ax.set_ylabel('Y',fontsize=16,linespacing=50.1)
    ax.set_zlabel('Z',fontsize=16,linespacing=50.1)
    # plt.colorbar(sc)
    ax.dist = 10.5
    plt.show()
    # plt.savefig(f'{sigma_path}/3d_visualization_norm_view{img_index}_model{t}.png', bbox_inches='tight')
    # plt.close()
    
    
    
    # plt.rcParams.update({'font.size': 12})
    # fig = plt.figure(figsize=(10,10))
    # ax = plt.axes(projection='3d')

    # sc = ax.scatter3D(datacoords_positive[:,0], datacoords_positive[:,1], datacoords_positive[:,2], c = datacolors_positive)
    # ax.set_xlim([datacoords_positive[:,0].min(),datacoords_positive[:,0].max()])
    # ax.set_ylim([datacoords_positive[:,1].min(),datacoords_positive[:,1].max()])
    # ax.set_zlim([datacoords_positive[:,2].min(),datacoords_positive[:,2].max()])
    # plt.title("test-img "+str(img_index)+" | Sigma Th.: "+str(th)+" | Timestep: "+str(t)+" | Number of 3D Points: "+str(len(datacoords_positive_norm)))
    # ax.set_xlabel('X',fontsize=16,linespacing=50.1)
    # ax.set_ylabel('Y',fontsize=16,linespacing=50.1)
    # ax.set_zlabel('Z',fontsize=16,linespacing=50.1)
    # # plt.colorbar(sc)
    # ax.dist = 10.5
    # plt.show()
    # # plt.savefig(f'{sigma_path}/3d_visualization_view{img_index}_model{t}.png', bbox_inches='tight')
    # # plt.close()
    # # break
    

    model_counter += 1

#%%

ri = []
for t, count in zip(t_range, range(0,len(t_range))):
    
    if t is not skip_time_steps:
        exp_t =datafolder+"/"+exp+str(t).zfill(3)
        
        exp_name0 = exp_t +"/"+ exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device0
        model_name0 = "model_"+exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device0+".pth"
        
        exp_name1 = exp_t +"/"+ exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device1
        model_name1 = "model_"+exp +"_"+ str(len(train_idx)) +"_cams_"+str(factor)+"factor_"+device1+".pth"
        
        
        model_path0 = "./"+exp_name0+"/"+model_name0+".pth"
        model_path1 = "./"+exp_name1+"/"+model_name1+".pth"
        
        
        if os.path.isfile(model_path0) | os.path.isfile(model_path1):
            ri.append(t)
        
        
#%%

n_samples = n_timesteps

#% create idct matrix operator
A = spfft.idct(np.identity(n_samples), norm='ortho', axis=0)
A = A[ri]
vx = cvx.Variable(n_samples)
objective = cvx.Minimize(cvx.norm(vx, 1))

# coeff_dataset = np.random.rand(len(ri), int(sum(non_neg_px)*3*sh_coeff_num) ) # for debugging

coeff_dataset_hat = np.empty((n_samples, 0) )


# do L1 optimization for each coefficient time-series
for coeff_time_series,count in zip(coeff_dataset.transpose(),range(0,coeff_dataset.shape[1])):
    if count%5000==0:
        print("time-series "+str(count)+"/"+str(coeff_dataset.shape[1]))
        
    constraints = [A@vx == coeff_time_series]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    
    #% reconstruct signal
    x = np.array(vx.value)
    x = np.squeeze(x)
    yhat = spfft.idct(x, norm='ortho', axis=0)
    ythat = spfft.dct(yhat, norm='ortho')
    
    coeff_dataset_hat = np.concatenate([coeff_dataset_hat, ythat[...,None]],axis=1)
    
    
sigma_dataset_hat = np.empty((n_samples, 0) )

# do L1 optimization for each coefficient time-seriesskip_time_steps = []

for sigma_time_series,count in zip(sigma_dataset.transpose(),range(0,sigma_dataset.shape[1])):
    if count%5000==0:
        print("time-series "+str(count)+"/"+str(sigma_dataset.shape[1]))
        
    constraints = [A@vx == sigma_time_series]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    
    #% reconstruct signal
    x = np.array(vx.value)
    x = np.squeeze(x)
    yhat = spfft.idct(x, norm='ortho', axis=0)
    ythat = spfft.dct(yhat, norm='ortho')
    
    sigma_dataset_hat = np.concatenate([sigma_dataset_hat, ythat[...,None]],axis=1)
    
# np.save("coeff_dataset_hat.npy", coeff_dataset_hat)
# np.save("coeff_dataset.npy", coeff_dataset)    
# np.save("sigma_dataset_hat.npy", sigma_dataset_hat)    
# np.save("sigma_dataset.npy", sigma_dataset)    


#%% Reconstruct all frames

coeff_dataset_hat = np.load("coeff_dataset_hat.npy")
coeff_dataset = np.load("coeff_dataset.npy")

sigma_dataset_hat = np.load("sigma_dataset_hat.npy")
sigma_dataset = np.load("sigma_dataset.npy")

#%%

t = np.arange(0, n_timesteps)

coeff = 1
px_sel = 10000
idx_vis = int(coeff*sh_coeff_num*10000)


plt.figure()
# plt.subplot(2,1,1)
# plt.plot(t[ri], coeff_dataset[:,idx_vis],color='b',linestyle='none',marker="o",markersize=4,label="Sampled Time steps")
plt.xlabel("Time Step")
# plt.subplot(2,1,2)
plt.plot(t, coeff_dataset_hat[:,idx_vis],color='k', label="Reconstructed timesteps using L1-norm")
# plt.xlim([0,300])
plt.title("Coefficient #" + str(coeff) + " for pixel #"+str(px_sel))
plt.legend()
plt.show()



# coeff_dataset.reshape(sum(non_neg_px),  3,int(sh_coeff_num))

#%%


th = 1.4


for img_index in range(0,nerf_obj.n_full_imgs):
    
    coeff_rec_t = coeff_dataset_hat[img_index,:]
    coeff_rec_t = coeff_rec_t.reshape(int(coeff_rec_t.shape[0]/(3*16)),3,int(sh_coeff_num))
    coeff_rec_t = coeff_rec_t[0:sum(non_neg_px),:,:]
    
    sigma_rec_t = sigma_dataset_hat[img_index,0:sum(non_neg_px)]
    
    # datasigma,dataweights,datacoords,datapx,datadisp,dataacc,datacolors,datacoeffs = pc_extraction(nerf_obj, nerf_obj.hn, nerf_obj.hf, nerf_obj.full_dataset, 
    #                                                                                             0, device, None,chunk_size=4, img_index=img_index, nb_bins=nb_bins, 
    #                                                                                             H=nerf_obj.H, W=nerf_obj.W,normalize=normalize_imgs)
    
    coeff_rec_t = torch.tensor(coeff_rec_t, device=device)
    sigma_rec_t = torch.tensor(sigma_rec_t, device=device)
    
    datasigma,dataweights,datacoords,datapx,datadisp,dataacc,datacolors,datacoeffs = pc_extraction_rec(nerf_obj, 
                                                                                                   coeff_rec_t.swapaxes(1, 2), sigma_rec_t,
                                                                                                   nerf_obj.hn, nerf_obj.hf, nerf_obj.full_dataset, 
                                                                                                0, device, None,chunk_size=4, img_index=img_index, nb_bins=nb_bins, 
                                                                                                H=nerf_obj.H, W=nerf_obj.W,normalize=normalize_imgs)
    
    
    break
    datacoeffs = datacoeffs.reshape(int(len(datacoeffs)/nb_bins), nb_bins, 3,int(sh_coeff_num))
    
    datasigma_filt = datasigma.max(axis=1)
    # non_neg_px = datasigma_filt>th
    datasigma_filt_idx = datasigma[non_neg_px].argmax(axis=1)
    datasigma_positive = datasigma[non_neg_px]
    
    datasigma_filt = datasigma_filt[non_neg_px]
    datacoords_positive = datacoords[non_neg_px,datasigma_filt_idx,:]
    datacolors_positive = datacolors[non_neg_px,datasigma_filt_idx,:]

    
    
    


    datacolors_positive = (1*(datacolors_positive - np.min(datacolors_positive))/np.ptp(datacolors_positive))
    
    datacoords_positive_norm = (1*(datacoords_positive - np.min(datacoords_positive))/np.ptp(datacoords_positive))

    #%
    

    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    sc = ax.scatter3D(datacoords_positive_norm[:,0], datacoords_positive_norm[:,1], datacoords_positive_norm[:,2], c = datacolors_positive)
    ax.set_xlim([0.,1])
    ax.set_ylim([0.,1])
    ax.set_zlim([0.,1])
    plt.title("test-img "+str(img_index)+" | Sigma Th.: "+str(th))
    ax.set_xlabel('X',fontsize=16,linespacing=50.1)
    ax.set_ylabel('Y',fontsize=16,linespacing=50.1)
    ax.set_zlabel('Z',fontsize=16,linespacing=50.1)
    # plt.colorbar(sc)
    ax.dist = 10.5
    plt.show()
    # plt.savefig(f'{sigma_path}/3d_visualization_norm_{img_index}.png', bbox_inches='tight')
    # plt.close()
    
    
    
    # plt.rcParams.update({'font.size': 12})
    # fig = plt.figure(figsize=(10,10))
    # ax = plt.axes(projection='3d')

    # sc = ax.scatter3D(datacoords_positive[:,0], datacoords_positive[:,1], datacoords_positive[:,2], c = datacolors_positive)
    # ax.set_xlim([datacoords_positive[:,0].min(),datacoords_positive[:,0].max()])
    # ax.set_ylim([datacoords_positive[:,1].min(),datacoords_positive[:,1].max()])
    # ax.set_zlim([datacoords_positive[:,2].min(),datacoords_positive[:,2].max()])
    # plt.title("test-img "+str(img_index)+" | Sigma Th.: "+str(th))
    # ax.set_xlabel('X',fontsize=16,linespacing=50.1)
    # ax.set_ylabel('Y',fontsize=16,linespacing=50.1)
    # ax.set_zlabel('Z',fontsize=16,linespacing=50.1)
    # # plt.colorbar(sc)
    # ax.dist = 10.5
    # plt.show()
    
    # plt.savefig(f'{sigma_path}/3d_visualization_{img_index}.png', bbox_inches='tight')
    # plt.close()
    
    
    break
    
    
    