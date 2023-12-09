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

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import albumentations
from matplotlib import pyplot as plt

class PitchSegmentationModel(nn.Module):
    def __init__(self):
        super(PitchSegmentationModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

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
        print(image_name)
        image_id = image_info["id"]
        keypoint_list = np.array([np.array(annotation["keypoint"])
                    for annotation in annotations['annotations']
                    if annotation['image_id'] == image_id], dtype=np.int32)
        # print(annotation_list)
        image_path = os.path.join(self.image_folder, image_name)
        image = cv2.cvtColor(cv2.imread(image_path, 
                                        cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

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


class PitchObjectDataset(Dataset):
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
        print(image_name)
        image_id = image_info["id"]
        keypoint_list = np.array([np.array(annotation["keypoint"])
                    for annotation in annotations['annotations']
                    if annotation['image_id'] == image_id], dtype=np.int32)
        # print(annotation_list)
        image_path = os.path.join(self.image_folder, image_name)
        image = cv2.cvtColor(cv2.imread(image_path, 
                                        cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

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