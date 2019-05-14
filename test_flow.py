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
from models.MonodepthModel import *
from models.PWC_net import *
from models.PWC_net import PWCDCNet
from utils.scene_dataloader import *
from utils.utils import *

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=256)
    parser.add_argument('--input_width',               type=int,   help='input width', default=512)
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', required=True)
    args = parser.parse_args()
    return args

args = get_args()

checkpoint = torch.load(args.checkpoint_path)
if args.model_name == 'monodepth':
    net = MonodepthNet().cuda()
elif args.model_name == 'pwc':
    net = pwc_dc_net().cuda()
    args.input_width = 832
net.load_state_dict(checkpoint['state_dict'])

former_test, latter_test, flow = get_flow_data(args.filenames_file, args.data_path)
TestFlowLoader = torch.utils.data.DataLoader(
        myImageFolder(former_test, latter_test, flow, args),
        batch_size = 1, shuffle = False, num_workers = 1, drop_last = False)

total_error = 0
fl_error = 0
num_test = 0
for batch_idx, (left, right, gt, mask, h, w) in enumerate(TestFlowLoader, 0):
    
    left_batch = torch.cat((left, torch.from_numpy(np.flip(left.numpy(), 3).copy())), 0)
    right_batch = torch.cat((right, torch.from_numpy(np.flip(right.numpy(), 3).copy())), 0)
    
    left = Variable(left_batch.cuda())
    right = Variable(right_batch.cuda())

    model_input = torch.cat((left, right), 1)
    if args.model_name == 'monodepth':
        disp_est_scale, disp_est = net(model_input)
    elif args.model_name == 'pwc':
        disp_est_scale = net(model_input)

    mask = np.ceil(np.clip(np.abs(gt[0,0]), 0, 1))

    disp_ori_scale = nn.UpsamplingBilinear2d(size=(int(h), int(w)))(disp_est_scale[0][:1])
    disp_ori_scale[0,0] = disp_ori_scale[0,0] * int(w) / args.input_width
    disp_ori_scale[0,1] = disp_ori_scale[0,1] * int(h) / args.input_height

    error, fl = evaluate_flow(disp_ori_scale[0].data.cpu().numpy(), gt[0].numpy(), mask.numpy())
    total_error += error
    fl_error += fl
    num_test += 1
    
total_error /= num_test
fl_error /= num_test
print("The average EPE is : ", total_error)
print("The average Fl is : ", fl_error)