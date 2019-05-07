import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from models.networks.submodules import *
from models.networks.resample2d_package.modules.resample2d import Resample2d

def gradient_x(img):
    gx = torch.add(img[:,:,:-1,:], -1, img[:,:,1:,:])
    return gx

def gradient_y(img):
    gy = torch.add(img[:,:,:,:-1], -1, img[:,:,:,1:])
    return gy

def get_disparity_smoothness(disp, pyramid):
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]
    
    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]
    
    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
    
    return smoothness_x + smoothness_y

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)
    
    #(input, kernel, stride, padding)
    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1, 0) - mu_x * mu_y
    
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    
    SSIM = SSIM_n / SSIM_d
    
    return torch.clamp((1 - SSIM) / 2, 0, 1)

def cal_grad2_error(flo, image, beta):
    """
    Calculate the image-edge-aware second-order smoothness loss for flo 
    """

    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy
    
    
    img_grad_x, img_grad_y = gradient(image)
    weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
    weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (torch.mean(beta*weights_x[:,:, :, 1:]*torch.abs(dx2)) + torch.mean(beta*weights_y[:, :, 1:, :]*torch.abs(dy2))) / 2.0

def warp_2(est, img, occ_mask, args):
    l1_warp2 = torch.abs(est - img) * occ_mask
    l1_reconstruction_loss_warp2 = torch.mean(l1_warp2) / torch.mean(occ_mask)
    ssim_warp2 = SSIM(est * occ_mask, img * occ_mask)
    ssim_loss_warp2 = torch.mean(ssim_warp2) / torch.mean(occ_mask)
    image_loss_warp2  = args.alpha_image_loss * ssim_loss_warp2 + (1 - args.alpha_image_loss) * l1_reconstruction_loss_warp2
    return image_loss_warp2

def create_mask(tensor, paddings):
    shape = tensor.shape
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner = Variable(torch.ones((inner_height, inner_width)).cuda())
    
    mask2d = nn.ZeroPad2d((paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]))(inner)
    mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
    mask4d = mask3d.unsqueeze(1)
    return mask4d.detach()

def create_border_mask(tensor, border_ratio = 0.1):
    num_batch, _, height, width = tensor.shape
    sz = np.ceil(height * border_ratio).astype(np.int).item(0)
    border_mask = create_mask(tensor, [[sz, sz], [sz, sz]])
    return border_mask.detach()

def length_sq(x):
    return torch.sum(x**2, 1, keepdim=True)

def create_outgoing_mask(flow):
    num_batch, channel, height, width = flow.shape
    
    grid_x = torch.arange(width).view(1, 1, width)
    grid_x = grid_x.repeat(num_batch, height, 1)
    grid_y = torch.arange(height).view(1, height, 1)
    grid_y = grid_y.repeat(num_batch, 1, width)
    
    flow_u, flow_v = torch.unbind(flow, 1)
    pos_x = grid_x.type(torch.FloatTensor) + flow_u.data.cpu()
    pos_y = grid_y.type(torch.FloatTensor) + flow_v.data.cpu()
    inside_x = (pos_x <= (width - 1)) & (pos_x >= 0.0)
    inside_y = (pos_y <= (height - 1)) & (pos_y >= 0.0)
    inside = inside_x & inside_y
    return inside.type(torch.FloatTensor).unsqueeze(1)

def get_mask(forward, backward, border_mask):
    flow_fw = forward
    flow_bw = backward
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    
    flow_bw_warped = Resample2d()(flow_bw, flow_fw)
    flow_fw_warped = Resample2d()(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh =  0.01 * mag_sq + 0.5
    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh).type(torch.cuda.FloatTensor)
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh).type(torch.cuda.FloatTensor)
    
    if border_mask is None:
        mask_fw = create_outgoing_mask(flow_fw)
        mask_bw = create_outgoing_mask(flow_bw)
    else:
        mask_fw = border_mask
        mask_bw = border_mask
    fw = mask_fw * (1 - fb_occ_fw)
    bw = mask_bw * (1 - fb_occ_bw)

    return fw, bw, flow_diff_fw, flow_diff_bw

def make_pyramid(image, num_scales):
    scale_image = [Variable(image.cuda())]
    height, width = image.shape[2:]

    for i in range(num_scales - 1):
        new = []
        for j in range(image.shape[0]):
            ratio = 2 ** (i+1)
            nh = height // ratio
            nw = width // ratio
            tmp = transforms.ToPILImage()(image[j]).convert('RGB')
            tmp = transforms.Resize([nh, nw])(tmp)
            tmp = transforms.ToTensor()(tmp)
            new.append(tmp.unsqueeze(0))
        this = torch.cat(new, 0)
        scale_image.append(Variable(this.cuda()))
        
    return scale_image

def evaluate_flow(flow, flow_gt, valid_mask=None):

    if valid_mask is None:
        tmp = np.multiply(flow_gt[0,:,:], flow_gt[1,:,:])
        valid_mask = np.ceil(np.clip(np.abs(tmp), 0, 1))
        
    N = np.sum(valid_mask)

    u = flow[0, :, :]
    v = flow[1, :, :]

    u_gt = flow_gt[0, :, :]
    v_gt = flow_gt[1, :, :]

    ### compute_EPE
    du = u - u_gt
    dv = v - v_gt

    du2 = np.multiply(du, du)
    dv2 = np.multiply(dv, dv)

    EPE = np.multiply(np.sqrt(du2 + dv2), valid_mask)
    EPE_avg = np.sum(EPE) / N
    
    ### compute FL
    bad_pixels = np.logical_and(
        EPE > 3,
        (EPE / np.sqrt(np.sum(np.square(flow_gt), axis=0)) + 1e-5) > 0.05)
    FL_avg = bad_pixels.sum() / valid_mask.sum()

    return EPE_avg, FL_avg