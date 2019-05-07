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

def evaluate_flow(flow, flow_gt, valid_mask=None):

    if valid_mask is None:
        tmp = np.multiply(flow_gt[0,:,:], flow_gt[1,:,:])
        valid_mask = np.ceil(np.clip(np.abs(tmp), 0, 1))
        

    N = max(np.sum(valid_mask), 1)

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
    
    threshold = np.maximum(3, np.sqrt(np.sum(np.square(flow_gt), axis=0)) * 0.05)
    outliers = (EPE > threshold)
    FL_avg = outliers.sum() / N
    
    return EPE_avg, FL_avg

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

def compute_flow():
    total_error = 0
    fl_error = 0
    noc_error = 0
    noc_fl_error = 0
    occ_error = 0
    occ_fl_error = 0
    for batch_idx, ((left, right, gt, mask, h, w), (_, _, _, mask_noc, _, _)) in enumerate(zip(TestFlowLoader, TestNocFlowLoader), 0):
        print(batch_idx)
        left_batch = torch.cat((left, torch.from_numpy(np.flip(left.numpy(), 3).copy())), 0)
        right_batch = torch.cat((right, torch.from_numpy(np.flip(right.numpy(), 3).copy())), 0)

        left = Variable(left_batch.cuda())
        right = Variable(right_batch.cuda())
        model_input = torch.cat((left, right), 1)
        disp_est_scale, disp_est = net(model_input)
        #disp_est_scale = net(model_input)
        #disp_est = [torch.cat((disp_est_scale[i][:,0,:,:].unsqueeze(1) / disp_est_scale[i].shape[3], disp_est_scale[i][:,1,:,:].unsqueeze(1) / disp_est_scale[i].shape[2]), 1) for i in range(4)]
        
        model_input_2 = torch.cat((right, left), 1)
        disp_est_scale_2, disp_est_2 = net(model_input_2)
        #disp_est_scale_2 = net(model_input_2)
        #disp_est_2 = [torch.cat((disp_est_scale_2[i][:,0,:,:].unsqueeze(1) / disp_est_scale_2[i].shape[3], disp_est_scale_2[i][:,1,:,:].unsqueeze(1) / disp_est_scale_2[i].shape[2]), 1) for i in range(4)]
        a = disp_est_scale[0][0,:,:,:].data.cpu().numpy()

        border_mask = create_border_mask(left, 0)
        fw, bw, diff_fw, diff_bw = get_mask(disp_est_scale[0], disp_est_scale_2[0], border_mask)
        m = nn.UpsamplingBilinear2d(size=(int(h), int(w)))(fw[:1])
        #plt.imshow(m.data.cpu().numpy()[0,0], cmap='gray')

        b = nn.UpsamplingBilinear2d(size=(int(h), int(w)))(disp_est_scale[0][:1])
        b[0,0] = b[0,0] * int(w) / 512
        b[0,1] = b[0,1] * int(h) / 256

        error, fl = evaluate_flow(b[0].data.cpu().numpy(), gt[0].numpy(), mask.numpy())
        total_error += error
        fl_error += fl
        
        error, fl = evaluate_flow(b[0].data.cpu().numpy(), gt[0].numpy(), mask_noc.numpy())
        noc_error += error
        noc_fl_error += fl
        
        error, fl = evaluate_flow(b[0].data.cpu().numpy(), gt[0].numpy(), (mask - mask_noc).numpy())
        occ_error += error
        occ_fl_error += fl
    total_error /= 200
    fl_error /= 200
    noc_error /= 200
    noc_fl_error /= 200
    occ_error /= 200
    occ_fl_error /= 200
    return total_error, fl_error, noc_error, noc_fl_error, occ_error, occ_fl_error