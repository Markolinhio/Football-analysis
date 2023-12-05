import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
plt.rcParams["figure.figsize"] = (24,18)
from cv_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision import utils as vutils
from torch.utils.data import random_split
import torch.optim.lr_scheduler as lr_scheduler

from shapely.geometry import Polygon
from matplotlib.patches import Polygon as pltPolygon
from PIL import Image
from tqdm import tqdm

from matplotlib import pyplot as plt
torch.cuda.empty_cache()
torch.cuda.is_available()


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
            
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

        
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
    
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x    
    
    
class PitchSegmentationModel(nn.Module):
    def __init__(self):
        super(PitchSegmentationModel, self).__init__()

        self.e1 = encoder_block(3, 16)
        self.e2 = encoder_block(16, 32)
        #self.e3 = encoder_block(32, 64)
        self.b = conv_block(32, 64)
        self.d1 = decoder_block(64, 32)
        self.d2 = decoder_block(32, 16)
        #self.d3 = decoder_block(32, 16)
        self.outputs = nn.Conv2d(16, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        #s3, p3 = self.e3(p2)
        b = self.b(p2)
        d1 = self.d1(b, s2)
        d2 = self.d2(d1, s1)
        #d3 = self.d3(d2, s1)
        outputs = self.outputs(d2)
        return outputs
    

class PitchDataset(Dataset):
    def __init__(self, image_folder, annotation_file, transform=None):
                        #image_transform=None, mask_transform=None):
        self.image_folder = image_folder
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform
        # self.image_transform = image_transform
        # self.mask_transform = mask_transform

    def __len__(self):
        return len(self.annotations['annotations'])

    def __getitem__(self, idx):
        image_info = self.annotations['images'][idx]
        image_name = image_info["file_name"]
        #print(image_name)
        image_id = image_info["id"]
        annotation_list = [annotation 
                           for annotation in self.annotations['annotations']
                           if annotation['image_id'] == image_id][0]
        # print(annotation_list)
        image_path = os.path.join(self.image_folder, image_name)
        image = cv2.cvtColor(cv2.imread(image_path, 
                                        cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

        annotation = np.array(annotation_list['segmentation'][0], 
                                dtype=np.int32).reshape((-1, 2))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [annotation], color=255)
        if len(annotation) == 0:
            plt.figure()
            plt.imshow(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        image = transforms.ToTensor()(image)
        #image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        mask = transforms.ToTensor()(mask)

        return image, mask
    

def iou_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou