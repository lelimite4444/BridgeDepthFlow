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

def get_kitti_cycle_data(file_path_train, path):
    f_train = open(file_path_train)
    former_left_image_train = list()
    latter_left_image_train = list()
    former_right_image_train = list()
    latter_right_image_train = list()
    
    for line in f_train:
        former_left_image_train.append(path+line.split()[0])
        latter_left_image_train.append(path+line.split()[2])
        former_right_image_train.append(path+line.split()[1])
        latter_right_image_train.append(path+line.split()[3])
        
    return former_left_image_train, latter_left_image_train, former_right_image_train, latter_right_image_train

def get_data(file_path_test, path):
    f_test = open(file_path_test)
    left_image_test = list()
    right_image_test = list()

    for line in f_test:
        left_image_test.append(path+line.split()[0])
        right_image_test.append(path+line.split()[1])
        
    return left_image_test, right_image_test

def get_flow_data(file_path_test, path):
    f_test = open(file_path_test)
    flow_test = list()
    former_image_test = list()
    latter_image_test = list()

    for line in f_test:
        former_image_test.append(path+line.split()[0])
        latter_image_test.append(path+line.split()[1])
        flow_test.append(path+line.split()[2])
        
    return former_image_test, latter_image_test, flow_test

def get_transform(param):
    return transforms.Compose([
        transforms.Resize([param.input_height, param.input_width]),
        transforms.ToTensor()
    ])
    
class myCycleImageFolder(data.Dataset):
    def __init__(self, left1, left2, right1, right2, training, param):
        self.right1 = right1
        self.left1 = left1
        self.right2 = right2
        self.left2 = left2
        self.training = training
        self.param = param
        
    def __getitem__(self, index):
        left1 = self.left1[index]
        right1 = self.right1[index]
        left2 = self.left2[index]
        right2 = self.right2[index]
        param = self.param
        left_image_1 = Image.open(left1).convert('RGB')
        right_image_1 = Image.open(right1).convert('RGB')
        left_image_2 = Image.open(left2).convert('RGB')
        right_image_2 = Image.open(right2).convert('RGB')

        
        #augmentation
        if self.training:
            
            #randomly flip
            if random.uniform(0, 1) > 0.5:
                left_image_1 = left_image_1.transpose(Image.FLIP_LEFT_RIGHT)
                right_image_1 = right_image_1.transpose(Image.FLIP_LEFT_RIGHT)
                left_image_2 = left_image_2.transpose(Image.FLIP_LEFT_RIGHT)
                right_image_2 = right_image_2.transpose(Image.FLIP_LEFT_RIGHT)
                
            #randomly shift gamma
            if random.uniform(0, 1) > 0.5:
                gamma = random.uniform(0.8, 1.2)
                left_image_1 = Image.fromarray(np.clip((np.array(left_image_1) ** gamma), 0, 255).astype('uint8'), 'RGB')
                right_image_1 = Image.fromarray(np.clip((np.array(right_image_1) ** gamma), 0, 255).astype('uint8'), 'RGB')
                left_image_2 = Image.fromarray(np.clip((np.array(left_image_2) ** gamma), 0, 255).astype('uint8'), 'RGB')
                right_image_2 = Image.fromarray(np.clip((np.array(right_image_2) ** gamma), 0, 255).astype('uint8'), 'RGB')
            
            #randomly shift brightness
            if random.uniform(0, 1) > 0.5:
                brightness = random.uniform(0.5, 2.0)
                left_image_1 = Image.fromarray(np.clip((np.array(left_image_1) * brightness), 0, 255).astype('uint8'), 'RGB')
                right_image_1 = Image.fromarray(np.clip((np.array(right_image_1) * brightness), 0, 255).astype('uint8'), 'RGB')
                left_image_2 = Image.fromarray(np.clip((np.array(left_image_2) * brightness), 0, 255).astype('uint8'), 'RGB')
                right_image_2 = Image.fromarray(np.clip((np.array(right_image_2) * brightness), 0, 255).astype('uint8'), 'RGB')
            
            #randomly shift color
            if random.uniform(0, 1) > 0.5:
                colors = [random.uniform(0.8, 1.2) for i in range(3)]
                shape = np.array(left_image_1).shape
                white = np.ones((shape[0], shape[1]))
                color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
                left_image_1 = Image.fromarray(np.clip((np.array(left_image_1) * color_image), 0, 255).astype('uint8'), 'RGB')
                right_image_1 = Image.fromarray(np.clip((np.array(right_image_1) * color_image), 0, 255).astype('uint8'), 'RGB')
                left_image_2 = Image.fromarray(np.clip((np.array(left_image_2) * color_image), 0, 255).astype('uint8'), 'RGB')
                right_image_2 = Image.fromarray(np.clip((np.array(right_image_2) * color_image), 0, 255).astype('uint8'), 'RGB')
                
        
        #transforms
        process = get_transform(param)
        left_image_1 = process(left_image_1)
        right_image_1 = process(right_image_1)
        left_image_2 = process(left_image_2)
        right_image_2 = process(right_image_2)
        
        return left_image_1, left_image_2, right_image_1, right_image_2
    def __len__(self):
        return len(self.left1)
    
class myImageFolder(data.Dataset):
    def __init__(self, left, right, flow, param):
        self.right = right
        self.left = left
        self.flow = flow
        self.param = param
        
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        param = self.param
        left_image = Image.open(left).convert('RGB')
        right_image = Image.open(right).convert('RGB')
      
        process = get_transform(param)
        left_image = process(left_image)
        right_image = process(right_image)
        
        if self.flow is not None:
            flow = self.flow[index]
            flow_image = cv2.imread(flow, -1)
            h, w, _ = flow_image.shape
            flo_img = flow_image[:,:,2:0:-1].astype(np.float32)
            invalid = (flow_image[:,:,0] == 0)

            flo_img = (flo_img - 32768) / 64
            flo_img[np.abs(flo_img) < 1e-10] = 1e-10
            flo_img[invalid, :] = 0

            f = torch.from_numpy(flo_img.transpose((2,0,1)))
            mask = torch.from_numpy((flow_image[:,:,0] == 1).astype(np.float32)).type(torch.FloatTensor)

            return left_image, right_image, f.type(torch.FloatTensor), mask, h, w
        
        return left_image, right_image
    
    def __len__(self):
        return len(self.left)