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
        if skip.size()[3] == x.size()[3] + 1:
            x = F.pad(x, (1, 0, 0, 0), "constant", 0)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x    
    
    
class PitchSegmentation(nn.Module):
    def __init__(self):
        super(PitchSegmentation, self).__init__()

        self.e1 = encoder_block(3, 16)
        self.e2 = encoder_block(16, 32)
        self.b = conv_block(32, 64)
        self.d1 = decoder_block(64, 32)
        self.d2 = decoder_block(32, 16)
        self.outputs = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        b = self.b(p2)
        d1 = self.d1(b, s2)
        d2 = self.d2(d1, s1)
        outputs = self.outputs(d2)
        return self.sigmoid(outputs)
    

class PitchObjectSegmentation(nn.Module):
    def __init__(self):
        super(PitchObjectSegmentation, self).__init__()

        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.b = conv_block(512, 1024)
        self.d4 = decoder_block(1024, 512)
        self.d3 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d1 = decoder_block(128, 64)
        self.outputs = nn.Conv2d(64, 6, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d4(b, s4)
        d2 = self.d3(d1, s3)
        d3 = self.d2(d2, s2)
        d4 = self.d1(d3, s1)
        outputs = self.outputs(d4)
        return self.softmax(outputs)   
    
    
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
        return len(self.annotations['images'])

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
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        image_info = self.annotations['images'][idx]
        image_name = image_info["file_name"]
        #print(image_name)
        image_id = image_info["id"]
        annotation_list = [annotation 
                           for annotation in self.annotations['annotations']
                           if annotation['image_id'] == image_id]
        # print(annotation_list)
        image_path = os.path.join(self.image_folder, image_name)
        image = cv2.cvtColor(cv2.imread(image_path, 
                                        cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        #print(annotation_list)
        for annotation in annotation_list:
            if annotation['category_id'] == 3:
                cv2.ellipse(mask, annotation['center'], annotation['length'],
                            annotation['angle'], 0, 360,
                            int(annotation['category_id'])*51, 5)
            elif annotation['category_id'] == 4:
                contours = np.array(annotation['segmentation']).reshape(1, -1, 2).astype(np.int32)
                cv2.drawContours(mask, contours, -1, 
                                int(annotation['category_id'])*51, -1)
            else:
                line = np.array(annotation['segmentation']).reshape(1, -1, 2).astype(np.int32)[0]
                cv2.polylines(mask, [line], False,
                              int(annotation['category_id'])*51, 5)
        # print(np.unique(mask))
        # mask = np.zeros(image.shape, dtype=np.uint8)
        # #print(annotation_list)
        # for annotation in annotation_list:
        #     if annotation['category_id'] == 3:
        #         cv2.ellipse(mask, annotation['center'], annotation['length'],
        #                     annotation['angle'], 0, 360, (0, 0, 255), 5)
        #     elif annotation['category_id'] == 4:
        #         contours = np.array(annotation['segmentation']).reshape(1, -1, 2).astype(np.int32)
        #         cv2.drawContours(mask, contours, -1, (0, 255, 0), -1)
        #     else:
        #         line = np.array(annotation['segmentation']).reshape(1, -1, 2).astype(np.int32)[0]

        #         if annotation['category_id'] == 1:
        #             color = (255, 0, 0)
        #         elif annotation['category_id'] == 2:
        #             color = (255, 255, 0)
        #         elif annotation['category_id'] == 5:
        #             color = (0, 255, 255)
        #         #if len(line) == 2:
        #             #cv2.line(mask, line[0], line[1], color, 5)
        #         #else:
        #         cv2.polylines(mask, [line], False, color, 5)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        image = transforms.ToTensor()(image)
        #image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        mask = transforms.ToTensor()(mask)
        return image, mask


def iou_loss(pred, target, smooth=1e-5):
    # pred = pred.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    # intersection = (pred & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    # union = (pred | target).float().sum((1, 2))         # Will be zzero if both are 0
    
    # iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0
    
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    # return 1 - thresholded 
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou


def iou_multiclass_loss(preds, targets, smooth=1e-5):
    # preds = torch.argmax(preds, dim=1).squeeze(1)
    # targets = (targets*255/51).squeeze(1)
    # preds = F.one_hot(preds.long(), 6).permute(3, 0, 1, 2)
    # targets = F.one_hot(targets.long(), 6).permute(3, 0, 1, 2)

    # class_pred = preds[1]
    # class_target = targets[1]
    # iou_sum = iou_loss(class_pred, class_target)
    # print(iou_sum)
    # for i in range(2, 5):
    #     class_pred = preds[i]
    #     class_target = targets[i]
    #     iou_sum += iou_loss(class_pred, class_target)
    
    # print(iou_sum)
    # return iou_sum/5

    # Flatten preds and targets
    preds = preds.view(-1, 6, preds.size(2), preds.size(3))
    targets = targets.view(-1, 1, targets.size(2), targets.size(3))

    iou_sum = 0.0
    total_present_classes = 0

    for batch_idx in range(preds.size(0)):
        # print("Batch id:",batch_idx)
        present_classes = (targets[batch_idx].unique() * 5).int() # Convert to long

        iou_target = 0.0
        for class_id in present_classes:
            if class_id == 0:  # Skip background class
                continue
            
            # Convert targets to one-hot encoding for the current class
            class_mask = (targets[batch_idx] == class_id/5).float()

            # Calculate intersection and union for the current class
            intersection = torch.sum(preds[batch_idx, class_id, :, :] * class_mask)
            union = torch.sum(preds[batch_idx, class_id, :, :] + class_mask) - intersection

            # Calculate IoU for the current class and add to the sum
            class_iou = (intersection + smooth) / (union + smooth)  # Adding a small epsilon to avoid division by zero
            # print("     Class id:",class_id.item(), ", Class iou:", class_iou.item())
            iou_target += (1 - class_iou)
            total_present_classes += 1
        iou_target = iou_target / (present_classes.size(0) - 1)
        # print("Image IoU:", iou_target.item())
        iou_sum += iou_target
    # Average over present classes and compute the mean IoU score
    iou = iou_sum / preds.size(0)
    #print("Total IoU:", iou.item())
    # print(iou)

    # # Convert targets to one-hot encoding
    # targets_one_hot = torch.zeros_like(preds)
    # targets_one_hot.scatter_(1, targets.long(), 1)

    # # Calculate intersection and union
    # intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))
    # union = torch.sum(preds + targets_one_hot, dim=(2, 3)) - intersection

    # # Calculate IoU for each class
    # iou_per_class = intersection / (union + 1e-6)  # Adding a small epsilon to avoid division by zero
    # print(iou_per_class)

    # # Average over classes and compute the mean IoU loss
    # mean_iou_loss = 1 - torch.mean(iou_per_class)

    return iou
