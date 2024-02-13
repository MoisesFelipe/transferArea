#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:40:59 2023

@author: felipe
"""



import torch, os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio
# import torchvision.transforms as T

from load_llff import load_llff_data


# import utils
import imageio.v2 as imageio

# import random
import glob
from PIL import Image
import cv2
import json
import progressbar
from time import sleep

# device = 'cuda:0'


class InfoNerfSH:
    
    def __init__(self, exp_name, exp, ext, train_idx, factor=1, device = 'cpu', num_val_img=5, llff=False, white_bkgd=False, half_res=False, normalize_imgs=True, spherify=False):
            
            self.exp_name = exp_name
            self.img_dir = "./"+exp
            self.exp = exp
            
            self.data_dir = self.img_dir + "/train" + "/*." + ext
            self.pos_filename = "transforms_train"
            self.device = device
            self.train_idx = train_idx
            
            # self.exp_name = exp+"_"+str(len(train_idx))+"imgs_"+str(factor)+"factor_nospherify_2ndtest"
            
            self.training_path = "./"+ self.exp_name + "/training_sh/"
            self.testing_path = "./"+ self.exp_name + "/testing_sh/"
            self.val_path = "./"+ self.exp_name + "/val_sh/"
            self.test_path = "./"+ self.exp_name + "/test_sh/"
            
            self.epoch = 0
            self.normalize_imgs = normalize_imgs
            self.llff = llff
            self.white_bkgd = white_bkgd
            self.half_res = half_res
            self.spherify = spherify
            self.num_val_img = num_val_img
            
            #% Directional info
            
            self.trans_t = lambda t : torch.as_tensor([
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,t],
                [0,0,0,1],
            ], dtype=torch.float32)
            
            self.rot_phi = lambda phi : torch.as_tensor([
                [1,0,0,0],
                [0,torch.cos(torch.as_tensor(phi)),-torch.sin(torch.as_tensor(phi)),0],
                [0,torch.sin(torch.as_tensor(phi)), torch.cos(torch.as_tensor(phi)),0],
                [0,0,0,1],
            ], dtype=torch.float32)
            
            self.rot_theta = lambda th : torch.as_tensor([
                [torch.cos(torch.as_tensor(th)),0,-torch.sin(torch.as_tensor(th)),0],
                [0,1,0,0],
                [torch.sin(torch.as_tensor(th)),0, torch.cos(torch.as_tensor(th)),0],
                [0,0,0,1],
            ], dtype=torch.float32)
    
    def pose_spherical(self,theta, phi, radius):
        c2w = self.trans_t(radius)
        c2w = self.rot_phi(phi/180.*np.pi) * c2w
        c2w = self.rot_theta(theta/180.*np.pi) * c2w
        c2w = torch.tensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) * c2w
        return c2w
    
    def load_imgs(self, data_dir,factor=0,cropping_factor_h=0,cropping_factor_w=0):
        img_names = sorted(glob.glob(data_dir),key=os.path.getmtime )
        # random.shuffle(img_names)
        # n_imgs = 30
        n_imgs = round(len(img_names))
        
        #% Loading images
        print("Loading images...")
        images = [np.asarray(Image.open(file)) for file in img_names]
        
        imagelist = []
        i=0
        for img in images:
            if img.shape[2]>3:
                img = img[:,:,0:3]
            else:
                img = img
            
            # rescaling factor
            if factor > 1:
                [halfres_h, halfres_w] = [hw // factor  for hw in img.shape[:2]]
                img = cv2.resize(
                    img, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                )
                
            if (cropping_factor_h>0) & (cropping_factor_w>0):
                
                path_img_crop = "./" + self.img_dir +"/" + self.exp_name + "/crop_"+ str(cropping_factor_h*100)+"h_" + str(cropping_factor_w*100) + "w"
                
                [h, w] = img.shape[:2]
                img = img[int(round(h*cropping_factor_h)):int(h-round(h*cropping_factor_h)),
                    int(round(w*cropping_factor_w)):int(w-round(w*cropping_factor_w))]
                
                # Check whether the specified path exists or not
                # if not os.path.exists(path_img_crop): # create it
                #     # Create a new directory because it does not exist
                #     os.makedirs(path_img_crop)
                #     print("Image directory has been created!")
                    
                # img_pil = Image.fromarray(img, "RGB")
                # img_pil.save(path_img_crop+"/m_" + str(i) +"." + ext)
            # plt.imshow(img)
            # plt.show()
            i+=1 
            
            imagelist.append(img)
            
        return imagelist
    
    def get_rays_np(self, H, W, focal, poses):
    
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
        rays_d = np.sum(dirs[..., np.newaxis, :] * poses[:3, :3], -1) # the first 3 x 3 block is in the camera’s point of view
        rays_o = np.broadcast_to(poses[:3, -1], np.shape(rays_d))
        
        return rays_o, rays_d
    
    def generate_rays(self, height, width, focal, pose):
        # Create a 2D rectangular grid for the rays corresponding to image dimensions
        i, j = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
        
        transformed_i = (i - width * 0.5) / focal # Normalize the x-axis coordinates
        transformed_j = -(j - height * 0.5) / focal # Normalize the y-axis coordinates
        k = -np.ones_like(i) # z-axis coordinates
        # Create the unit vectors corresponding to ray directions
        directions = np.stack([transformed_i, transformed_j, k], axis=-1)
        
        # Compute Origins and Directions for each ray
        camera_directions = directions[..., None, :] * pose[:3, :3]
        ray_directions = np.einsum("ijl,kl", directions, pose[:3, :3])
        ray_origins = np.broadcast_to(pose[:3, -1], ray_directions.shape)
        
        # return np.stack([ray_origins, ray_directions])
        return ray_origins, ray_directions
    
    def get_poses(self, images,img_dir,filename,normalize=False):
        
        n_imgs = round(len(images))
        
        rows,col,ch = images[0].shape
        
        _, poses, render_poses, hwf, _ = self.load_blender_data(img_dir, "train")
        
        H, W, focal = hwf
        
        # Loading positions
        print("Loading camera positions and reshaping input images...")
        
        rays = [self.get_rays_np(rows, col, focal, p) for p in poses[:, :3, :4]]
        dataset = np.empty([0,9])
        with progressbar.ProgressBar(max_value=n_imgs) as bar:
            for img, ray, i in zip(images, rays, range(0,n_imgs)):
                if normalize:
                    img = img.reshape((rows*col,3))/255
                else:
                    img = img.reshape((rows*col,3))
                    
                ray_o = ray[0].reshape((rows*col,3))
                ray_d = ray[1].reshape((rows*col,3))
                rays_full = np.concatenate((ray_o,ray_d,img),1)
                dataset = np.append(dataset, rays_full,axis=0)
                
                
                sleep(0.0000001)
                bar.update(i+1)
                
        return torch.tensor(dataset, dtype=torch.float32).to(self.device)
    
    #% Main NeRF Functions
    
    #% SH Evaluation
    
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]
    
    def eval_sh_coeff(self, deg, sh, dirs):
        
        C = sh.shape[-2]
        # print("C_sh_shape: "+str(C))
        C0 = self.C0
        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        C4 = self.C4
        
        result = C0 * sh[..., 0]
        if deg > 0:
            # x, y, z = dirs[..., 0:1].reshape(len(dirs)), dirs[..., 1:2].reshape(len(dirs)), dirs[..., 2:3].reshape(len(dirs))
            x, y, z = dirs[..., 0:1, None], dirs[..., 1:2, None], dirs[..., 2:3, None]
            # print("y: "+str(y.shape))
            result = (result -
                    C1 * y * sh[..., 1] +
                    C1 * z * sh[..., 2] -
                    C1 * x * sh[..., 3])
            if deg > 1:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result = (result +
                        C2[0] * xy * sh[..., 4] +
                        C2[1] * yz * sh[..., 5] +
                        C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                        C2[3] * xz * sh[..., 7] +
                        C2[4] * (xx - yy) * sh[..., 8])
    
                if deg > 2:
                    result = (result +
                            C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                            C3[1] * xy * z * sh[..., 10] +
                            C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                            C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                            C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                            C3[5] * z * (xx - yy) * sh[..., 14] +
                            C3[6] * x * (xx - 3 * yy) * sh[..., 15])
                    if deg > 3:
                        result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                                C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                                C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                                C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                                C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                                C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                                C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                                C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                                C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
         
        return result.squeeze(2)
    
    def compute_psnr(self, mse):
        return -10.0 * np.log(mse) / np.log(10.0)
    
    def loss_func_sh(self, rgb_raw, deg, sh, dirs):
        rgb_pred = self.eval_sh_coeff(deg, sh, dirs)
        return ((rgb_pred - rgb_raw) ** 2).mean(), rgb_pred
    
    def load_blender_data(self, basedir, white_bkgd=False, half_res=False, testskip=1):
        splits = ['train', 'val', 'test']
        # splits = [splits]
        metas = {}
        for s in splits:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)
        
            
        all_imgs = []
        all_poses = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []
            if s=='train' or testskip==0:
                skip = 1
            else:
                skip = testskip
                
            for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, frame['file_path'] + '.png')
                img = imageio.imread(fname)
                # print("antes: "+str(np.shape(img)))
                if half_res:
                   img = cv2.resize(img, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
                # print("depois: "+str(np.shape(img)))
                imgs.append(img)
                poses.append(np.array(frame['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            
            
            
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
        
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
        # i_split = [np.arange(counts[i], counts[i+1]) for i in range(1)]
        
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        
        H, W = imgs[0].shape[:2]
        
        if half_res:
            W *=2
            H *=2
            
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        render_poses = torch.stack([self.pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
        
        if white_bkgd:
            imgs = imgs[..., :3]*imgs[..., -1:] + (1.-imgs[..., -1:])
        else:
            imgs = imgs[..., :3]
        
        if half_res:
            H = H//2
            W = W//2
            focal = focal/2.
           
        return imgs, poses, render_poses, [H, W, focal], i_split
    
    @torch.no_grad()
    def dataset_plot(self, hn, hf, dataset, device,path_imgs, normalize=False, H=400, W=400):
        
        # Check whether the specified path exists or not
        if not os.path.exists(path_imgs): # create it
            # Create a new directory because it does not exist
            os.makedirs(path_imgs)
            print("Image directory has been created!")
        
        plt.ioff()
        n_imgs = int(len(dataset)/(H*W))
        for i in range(0,n_imgs):
            ray_color = dataset[i * H * W: (i + 1) * H * W, 6:9]
            img = ray_color.cpu().numpy().reshape(H, W, 3)
            if normalize:
                img *= 255
            plt.figure()
            plt.imshow(img.astype(int))
            # plt.show()
            plt.savefig(f'{path_imgs}/img_orig_train_{i}.png', bbox_inches='tight')
            plt.close()
    
    @torch.no_grad()
    def train_plot(self, model, hn, hf, dataset, device, path_imgs, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
        
        # Check whether the specified path exists or not
        if not os.path.exists(path_imgs): # create it
            # Create a new directory because it does not exist
            os.makedirs(path_imgs)
            print("Image directory has been created!")
            
        ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
        ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    
        data = []
        for i in range(int(np.ceil(H / chunk_size))):
            ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
    
            regenerated_px_values,_,_,_,_,_,_,_,_ = self.render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
            data.append(regenerated_px_values)
            
            
        img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
        plt.figure()
        plt.imshow(img.astype(int))
        plt.savefig(f'{path_imgs}/img_render_train_{img_index}.png', bbox_inches='tight')
        plt.close()
        
        img = (255*(img - np.min(img))/np.ptp(img)).astype(int)   
        
        plt.figure()
        plt.imshow(img)
        plt.savefig(f'{path_imgs}/img_render_train_{img_index}_normalized.png', bbox_inches='tight')
        plt.close()
        
        
    @torch.no_grad()
    def val(self, model, hn, hf, dataset, device, epoch, path_imgs, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400,normalize=False):
        model.eval()
        # Check whether the specified path exists or not
        if not os.path.exists(path_imgs): # create it
            # Create a new directory because it does not exist
            os.makedirs(path_imgs)
            print("Image directory has been created!")
            
        ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
        ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
        
        datapx,datadisp,dataacc = [],[],[]
        for i in range(int(np.ceil(H / chunk_size))):
            ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
    
            regenerated_px_values,_,disp,acc,_,_,_,_,_ = self.render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
            datapx.append(regenerated_px_values)
            datadisp.append(disp.repeat(1,3) )
            dataacc.append(acc.repeat(1,3))
        
        plt.ioff()
        # img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
        # saving rgb
        img = torch.cat(datapx).data.cpu().numpy().reshape(H, W, 3)
        if normalize:
            # img *= 255
            img = (255*(img - np.min(img))/np.ptp(img)).astype(int)   
        plt.figure()
        plt.imshow(img)
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
        
        
    # val(model, 2, 6, testing_dataset, device, epoch, val_path, img_index=np.random.randint(0,n_test_imgs ), nb_bins=192, H=400, W=400)
        
    @torch.no_grad()
    def test(self, model, hn, hf, dataset, epoch, device, path_imgs, chunk_size=512, img_index=0, nb_bins=192, H=400, W=400,normalize=False):
        model.eval()
        # Check whether the specified path exists or not
        if not os.path.exists(path_imgs): # create it
            # Create a new directory because it does not exist
            os.makedirs(path_imgs)
            print("Image directory has been created!")
            
        ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
        ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    
        datapx,datadisp,dataacc = [],[],[]
        for i in range(int(np.ceil(H / chunk_size))):
            ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
    
            regenerated_px_values,_,disp,acc,_,_ = self.render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
            datapx.append(regenerated_px_values )
            datadisp.append(disp.repeat(1,3) )
            dataacc.append(acc.repeat(1,3))
        
        plt.ioff()
        # saving rgb
        img = torch.cat(datapx).data.cpu().numpy().reshape(H, W, 3)
        # img = to8b(torch.cat(datapx).data.cpu().numpy()).reshape(H, W, 3)
        if normalize:
            # img *= 255
            img = (255*(img - np.min(img))/np.ptp(img)).astype(int)   
        plt.figure()
        plt.imshow(img)
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
        
    
    @torch.no_grad()
    def test_param(self, model, hn, hf, dataset, epoch, device, path_imgs, chunk_size=512, img_index=0, nb_bins=192, H=400, W=400,normalize=False):
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
        
        datacoeffs = np.empty((0,3,int(self.sh_coeff_num)))
        
        plt.ioff()
        for i in range(int(np.ceil(H / chunk_size))):
            ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
    
            regenerated_px_values,regularization,disp,acc,weights,sigma, coords,colors,sh_coeffs = self.render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
            
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
    def test_side_by_side(self, model, hn, hf, dataset, epoch, device, path_imgs, chunk_size=512, img_index=0, nb_bins=192, H=400, W=400,normalize=False):
        model.eval()
        # Check whether the specified path exists or not
        if not os.path.exists(path_imgs): # create it
            # Create a new directory because it does not exist
            os.makedirs(path_imgs)
            print("Image directory has been created!")
            
        ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
        ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
        img_orig = dataset[img_index * H * W: (img_index + 1) * H * W, 6:9].data.cpu().numpy().reshape(H, W, 3)
        
        plt.ioff()
        datapx,datadisp,dataacc = [],[],[]
        for i in range(int(np.ceil(H / chunk_size))):
            ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
    
            regenerated_px_values,_,disp,acc,_,_,_,_,_ = self.render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
            
            # regenerated_px_values,regularization,disp,acc,weights,sigma, coords,colors,sh_coeffs = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
            
            datapx.append(regenerated_px_values )
            datadisp.append(disp.repeat(1,3) )
            dataacc.append(acc.repeat(1,3))
            
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
    
    
    class NerfModel(nn.Module):
        def __init__(self, embedding_dim_pos=20, embedding_dim_direction=8, hidden_dim=128, sh_coeff=16):
            # super(NerfModel, self).__init__()
            super(self.__class__, self).__init__()
    
            self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 3, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
    
            self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 3 + hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim + 1), )
    
            self.block3 = nn.Sequential(nn.Linear(embedding_dim_direction * 3 + hidden_dim, hidden_dim // 2), nn.ReLU(), )
            # self.block3 = nn.Sequential(nn.Linear( hidden_dim, hidden_dim // 2), nn.ReLU(), )
            self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, int(sh_coeff)*3), nn.ReLU(), )
            self.block5 = nn.Sequential(nn.Unflatten(1,(int(sh_coeff),3)), nn.Sigmoid(), ) 
            
            self.embedding_dim_pos = embedding_dim_pos
            self.embedding_dim_direction = embedding_dim_direction
            self.relu = nn.ReLU()
    
        @staticmethod
        def positional_encoding(x, L):
            out = torch.empty(x.shape[0], x.shape[1] * 2 * L, device=x.device)
            for i in range(x.shape[1]):
                for j in range(L):
                    out[:, i * (2 * L) + 2 * j] = torch.sin(2 ** j * x[:, i])
                    out[:, i * (2 * L) + 2 * j + 1] = torch.cos(2 ** j * x[:, i])
            return out
    
        
        def forward(self, o, d):
            emb_x = self.positional_encoding(o, self.embedding_dim_pos // 2)
            emb_d = self.positional_encoding(d, self.embedding_dim_direction // 2)
            h = self.block1(emb_x)
            tmp = self.block2(torch.cat((h, emb_x), dim=1))
            h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
            h = self.block3(torch.cat((h, emb_d), dim=1)) # concatenating with direction info
            # h = self.block3(h)
            h2 = self.block4(h)
            # c = c.reshape(c.shape[0], 3,c.shape[1]//3)
            # print("h: "+str(h2.shape))
            c = self.block5(h2)
            # print("c: "+str(c.shape))
            return c, sigma
    
    
    def compute_accumulated_transmittance(self, alphas):
        accumulated_transmittance = torch.cumprod(alphas, 1)
        return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                          accumulated_transmittance[:, :-1]), dim=-1)
    
    
    def render_rays(self, nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192, T=0.1):
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
        
        # print("x: "+str(ray_directions.device) )
        # print("x: "+str(x.device))
        # print("model: "+str(next(nerf_model.parameters()).device))
        sh_coeffs, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
        
        # evaluating SH function
        sh_coeffs = sh_coeffs.swapaxes(1, 2)
        colors = self.eval_sh_coeff(self.sh_order, sh_coeffs.reshape(
            *sh_coeffs.shape[:-1],
            -1,
            (self.sh_order + 1) ** 2), ray_directions.reshape(-1, 3))
            
        # print("color: "+str(colors.shape))
        colors = colors.reshape(x.shape)
        sigma = sigma.reshape(x.shape[:-1])
        # print("colors_reshape: "+str(colors.shape))
        
        alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
        weights = self.compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    
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
    
    def render_sh_coeff(self, sh_coeffs, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192, T=0.1):
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
        
        sh_coeffs, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
        
        # evaluating SH function
        sh_coeffs = sh_coeffs.swapaxes(1, 2)
        colors = self.eval_sh_coeff(self.sh_order, sh_coeffs.reshape(
            *sh_coeffs.shape[:-1],
            -1,
            (self.sh_order + 1) ** 2), ray_directions.reshape(-1, 3))
            
        # print("color: "+str(colors.shape))
        colors = colors.reshape(x.shape)
        sigma = sigma.reshape(x.shape[:-1])
        # print("colors_reshape: "+str(colors.shape))
        
        alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
        weights = self.compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    
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
    
    def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)
    
    def create_dir(path_imgs):
        # Check whether the specified path exists or not
        if not os.path.exists(path_imgs): # create it
            # Create a new directory because it does not exist
            os.makedirs(path_imgs)
            print("Image directory has been created!")
            
    @torch.no_grad()
    def render_path(self, render_poses, model, device, batch_size, hwf, hn, hf, nb_bins, T, savedir=None, render_factor=0):
        
        model.eval()
        
        if savedir is not None:
            self.create_dir(savedir)
        
            
        H, W, focal = hwf
        # device = next(model.parameters()).device
        model.eval()
        model.to(device)
        
        if render_factor != 0:
            # Render downsampled for speed
            H = H//render_factor
            W = W//render_factor
            focal = focal/render_factor
            
        rgbs = np.empty([0,3])
        disps = np.empty([0,1])
        accs = np.empty([0,1])
        
        # batch_start = torch.arange(0, len(render_poses),1)
        # with tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            
        for i, c2w in enumerate(render_poses):
            # print("rendering image "+str(i))
            # rays_o, rays_d = get_rays_np(H, W, focal, c2w.detach().numpy() )
            rays_o, rays_d = self.get_rays_np(H, W, focal, c2w )
            
            rays_o = torch.tensor(rays_o.reshape((H*W,3)), dtype=torch.float32)
            rays_d = torch.tensor(rays_d.reshape((H*W,3)), dtype=torch.float32)
            
            rays = torch.cat( (rays_o,rays_d),dim=1).to("cpu")
            
            # print("batch: "+str(rays.shape))
            # print("batch-size: "+str(batch_size))
            # data_loader = DataLoader(rays, batch_size=batch_size, shuffle=False)
            
            rgb = torch.empty(0, 3,device="cpu")
            acc = torch.empty(0, 1,device="cpu")
            disp = torch.empty(0, 1,device="cpu")
            
            batch_size_ini = batch_size
            
            # batch_count = 0
            # for batch, start in zip(tqdm(data_loader, disable=True),bar):
                
                # bar.set_description(f"Rendering_image {i}")
            print("test-image "+str(i))
            for batch_count in range(0,int((H*W)/batch_size) +1):
                if (((batch_count+1)*batch_size)>(H*W)):
                    batch_size = H*W-len(rgb)
                    
                ray_origins = rays[batch_count*batch_size:(batch_count+1)*batch_size,:3]
                ray_directions = rays[batch_count*batch_size:(batch_count+1)*batch_size,3:6]
                ray_origins = ray_origins.to(device)
                ray_directions = ray_directions.to(device)
                
                # print(ray_origins.shape)
                # print(ray_directions.shape)
                rgb_aux, _, disps_aux, acc_aux,_,_ = self.render_rays(model, ray_origins, ray_directions, hn=hn, hf=hf,
                                                                    nb_bins=nb_bins, T=T)
                
                
                rgb = torch.cat((rgb,rgb_aux.to("cpu") ),dim=0)
                disp = torch.cat((disp,disps_aux.to("cpu")),dim=0)
                acc = torch.cat((acc,acc_aux.to("cpu")),dim=0)
            
                # print("batch "+str(batch_count)+"/"+str(int((H*W)/batch_size)))
                
                del ray_origins,ray_directions
                
                if batch_size is not batch_size_ini:
                    batch_size = batch_size_ini
            
            
            
            rgbs = np.append(rgbs, rgb.cpu().detach().numpy(),axis=0)
            disps = np.append(disps, disp.cpu().detach().numpy(),axis=0)
            accs = np.append(accs, acc.cpu().detach().numpy(),axis=0)
            
            # datapx.append(regenerated_px_values )
            # datadisp.append(disp.repeat(1,3) )
            # dataacc.append(acc.repeat(1,3))
            
                # batch_count+=1
                
                
            # rgbs.append(rgb.detach().numpy())
            # disps.append(disp.detach().numpy())
            # accs.append(acc.detach().numpy())
            
            # print("rgbs: "+str(rgbs.shape))
            
            if i == 0:
                print(rgb.shape, disp.shape, accs.shape)
    
    
            if savedir is not None:
                # rgb8 = to8b(rgbs[-1]).reshape((H,W,3))
                # rgb8 = to8b(rgbs[-1])
                
                # print("rgb8: "+str(rgb8.shape))
                
                filename_rgb = os.path.join(self.test_path, '{:03d}rgb.png'.format(i))
                filename_disp = os.path.join(self.test_path, '{:03d}disp.png'.format(i))
                filename_acc = os.path.join(self.test_path, '{:03d}acc.png'.format(i))
                
                # saving rgb
                img = rgb.data.cpu().numpy().reshape(H, W, 3)
                plt.figure()
                # plt.imshow(img.astype(int))
                plt.imshow(img)
                plt.savefig(filename_rgb, bbox_inches='tight')
                plt.close()
                
                # saving disparity
                disp_img = disp.data.cpu().numpy().reshape(H, W, 1).repeat(3,2)
                plt.figure()
                # plt.imshow(disp_img.astype(int))
                plt.imshow(disp_img)
                plt.savefig(filename_disp, bbox_inches='tight')
                plt.close()
                
                # saving depth
                depth_img = acc.data.cpu().numpy().reshape(H, W, 1).repeat(3,2)
                plt.figure()
                # plt.imshow(depth_img.astype(int))
                plt.imshow(depth_img)
                plt.savefig(filename_acc, bbox_inches='tight')
                plt.close()
                
                
    
        rgbs = np.stack(rgbs, 0)
        accs = np.stack(accs, 0)
        disps = np.stack(disps, 0)
    
        return rgbs, disps, accs
    
    

        
    def load_pre_process_data(self, llff, white_bkgd, half_res, hn=2., hf=6.):
        
        if llff:
            self.imagelist, self.poses, self.bds, self.render_poses, self.i_test = load_llff_data(self.img_dir, self.factor,spherify=self.spherify)
            self.hwf = self.poses[0, :3, -1]
            self.poses = self.poses[:, :3, :4]
            self.render_poses = self.render_poses[:, :3, :4]
            
            if not isinstance(self.i_test, list):
                self.i_test = [self.i_test]
        
            self.i_val = self.i_test
            self.i_train = np.array([i for i in np.arange(int(self.imagelist.shape[0])) if
                                        (i not in self.i_test and i not in self.i_val)])
            self.bds = torch.as_tensor(self.bds)
            self.hn = torch.min(self.bds) * .9
            self.hf = torch.max(self.bds) * 1.
        else:
            self.imagelist, self.poses, self.render_poses, self.hwf, self.i_split = self.load_blender_data(self.img_dir, white_bkgd, half_res=half_res)
            self.i_train, self.i_val, self.i_test = self.i_split
            
            self.hn = hn
            self.hf = hf
            
            
        self.H, self.W, self.focal = self.hwf
        self.H, self.W = int(self.H), int(self.W)
        self.hwf = [self.H, self.W, self.focal]
        self.resolution = self.H*self.W


    def getting_rays(self):
        
        self.rays = [self.get_rays_np(self.H, self.W, self.focal, p) for p in self.poses[:, :3, :4]]
        self.rays = np.stack(self.rays, axis=0)  # [N, ro+rd, H, W, 3]

        self.rays_rgb = np.concatenate([self.rays, self.imagelist[:, None, ...]], 1)
        self.rays_rgb = self.rays_rgb.swapaxes(1, 2).swapaxes(2, 3)
        self.rays_rgb = self.rays_rgb.reshape((self.H*self.W*len(self.imagelist),9))
        
        
        self.full_dataset = np.stack([ self.rays_rgb[i*self.H*self.W:(i+1)*self.H*self.W] for i in range(0,len(self.poses)-1)], axis=0)  # train images only
        self.full_dataset = self.full_dataset.reshape((self.H*self.W*(len(self.poses)-1),9))
        self.full_dataset = self.full_dataset.astype(np.float32)
        self.full_dataset = torch.tensor(self.full_dataset, dtype=torch.float32).to("cpu")


        # training_dataset = np.stack([ rays_rgb[i*H*W:(i+1)*H*W] for i in i_train], axis=0)  # train images only
        # training_dataset = training_dataset.reshape((H*W*len(i_train),9))
        self.training_dataset = np.stack([ self.rays_rgb[i*self.H*self.W:(i+1)*self.H*self.W] for i in self.train_idx], axis=0)  # train images only
        self.training_dataset = self.training_dataset.reshape((self.H*self.W*len(self.train_idx),9))
        self.training_dataset = self.training_dataset.astype(np.float32)
        self.training_dataset = torch.tensor(self.training_dataset, dtype=torch.float32).to(self.device)


        self.val_dataset = np.stack([ self.rays_rgb[i*self.H*self.W:(i+1)*self.H*self.W] for i in self.i_val], axis=0)  # validation images only
        self.val_dataset = self.val_dataset.reshape((self.H*self.W*len(self.i_val),9))
        self.val_dataset = self.val_dataset.astype(np.float32)
        self.val_dataset = torch.tensor(self.val_dataset, dtype=torch.float32).to(self.device)


        self.testing_dataset = np.stack([ self.rays_rgb[i*self.H*self.W:(i+1)*self.H*self.W] for i in self.i_test], axis=0)  # test images only
        self.testing_dataset = self.testing_dataset.reshape((self.H*self.W*len(self.i_test),9))
        self.testing_dataset = self.testing_dataset.astype(np.float32)
        self.testing_dataset = torch.tensor(self.testing_dataset, dtype=torch.float32).to(self.device)
        
        self.n_test_imgs = int(len(self.testing_dataset)/(self.H*self.W))
        self.n_train_imgs = int(len(self.training_dataset)/(self.H*self.W))
        self.n_val_imgs = int(len(self.val_dataset)/(self.H*self.W))
        self.n_full_imgs = int(len(self.full_dataset)/(self.H*self.W))

        # plot training images
        # self.dataset_plot(self.hn, self.hf, self.training_dataset, device, self.training_path, normalize=True, H=self.H, W=self.W)
        
    def train(self):
        
        self.model.to(self.device)
        
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        loss_fn = nn.MSELoss()  # mean square error
        psnr_fn = PeakSignalNoiseRatio()
        training_loss = []
        
        batch_start = torch.arange(0, len(self.training_dataset), self.batch_size)
        
        for self.epoch in range(self.epoch_init, self.nb_epochs):
            with tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
                
                self.model.train()
                
                
                batch_count = 0.0
                loss_acc = 0
                mse_acc = 0
                psnr_acc = 0
                reg_acc = 0
                
                before_lr = self.model_optimizer.param_groups[0]["lr"]
                
                data_loader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)
                
                
                for batch, start in zip(tqdm(data_loader, disable=True),bar):
                    batch_count+=1.0
                    bar.set_description(f"Epoch {self.epoch}")
                    ray_origins = batch[:, :3].to(self.device)
                    ray_directions = batch[:, 3:6].to(self.device)
                    ground_truth_px_values = batch[:, 6:].to(self.device)
                    
                    regenerated_px_values, regularization, disp, acc,weights, sigma,_,_,_ = self.render_rays(self.model, ray_origins, ray_directions, hn=self.hn, hf=self.hf,
                                                                        nb_bins=self.nb_bins, T=self.T)
                    
                    # resetting nerf model
                    if regularization==0.0:
                        print("resetting the model...")
                        self.nerf_setting()
                        
                    loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
                    # loss = loss_fn(ground_truth_px_values, regenerated_px_values)
                    training_loss.append(loss.item())
                    mse_acc += float(loss)
                    loss = loss + self.lambda_mul * regularization
                    
                    psnr = float(psnr_fn(ground_truth_px_values.to("cpu"), regenerated_px_values.to("cpu")) )
                    psnr_acc += psnr
                    
                    reg_acc += float(self.lambda_mul * regularization)
                    
                    self.model_optimizer.zero_grad()
                    loss.backward()
                    self.model_optimizer.step()
                    loss_acc += float(loss)
                    
                    # print progress
                    bar.set_postfix(loss=float(loss_acc/batch_count),mse=float(mse_acc/batch_count),reg=float(reg_acc/batch_count), psnr=float(psnr_acc/batch_count), lr=float(before_lr))

                # validation_img
                if self.epoch%self.num_val_img==0:
                    self.val(self.model, self.hn, self.hf, self.val_dataset, 
                             self.device, self.epoch, self.val_path, chunk_size=10, 
                             img_index=1, nb_bins=self.nb_bins, 
                             H=self.H, W=self.W, normalize=self.normalize_imgs)
                    
                    self.val(self.model, self.hn, self.hf, self.val_dataset, 
                             self.device, self.epoch, self.val_path, chunk_size=10, 
                             img_index=0, nb_bins=self.nb_bins, 
                             H=self.H, W=self.W, normalize=self.normalize_imgs)
                    
            if self.scheduler_freq is not None:
                if (self.scheduler_freq % self.epoch):
                    print("atualizando scheduler")
                    self.scheduler.step()
                    
    
    def render_test_imgs(self, chunk_size=4):
        
        for img_index in range(self.n_full_imgs):
            if sum(self.train_idx==img_index)==1:
                self.test_side_by_side(self.model, self.hn, self.hf, 
                                       self.full_dataset, self.epoch, self.device, 
                                       self.test_path, chunk_size=chunk_size, 
                                       img_index=img_index, nb_bins=self.nb_bins, 
                                       H=self.H, W=self.W,normalize=self.normalize_imgs)

                
    def nerf_setting(self):
        
        #% Setting NeRF Model
        self.model = self.NerfModel(hidden_dim=self.hidden_dim, sh_coeff=self.sh_coeff_num).to(self.device)
        
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        
        
        if self.scheduler_freq is not None:
            
            if self.scheduler_freq<=0:
                self.scheduler_freq = None
                    
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer, 
                                                                  milestones=np.linspace(0, self.nb_epochs, self.n_milestones_steps).astype(int).tolist(), 
                                                                  gamma=self.gamma)
            

    def __run__(self, device, hidden_dim=256, sh_order=3, sh_coeff_num=16, 
                batch_size = int(256*4), lr=7e-5, nb_epochs=1000,
                n_milestones_steps = 50,gamma=0.5,nb_bins=192,T=0.1,lambda_mul=4,scheduler_freq=1):
        
        self.load_pre_process_data(self.llff, self.white_bkgd, self.half_res)
        self.getting_rays()
        
        # training params
        self.epoch_init = 0
        self.epoch = self.epoch_init
        self.device = device
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.sh_coeff_num = sh_coeff_num
        self.nb_bins= nb_bins
        self.T = T
        self.lr = lr
        self.lambda_mul = lambda_mul
        self.sh_order = sh_order 
        self.scheduler = None
        self.gamma = gamma
        self.n_milestones_steps = n_milestones_steps
        self.scheduler_freq = scheduler_freq

        self.nerf_setting()
        