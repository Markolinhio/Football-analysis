import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

from pathlib import Path
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import yaml
import shutil
from datetime import date
plt.rcParams["figure.figsize"] = (24,18)

from sklearn.model_selection import train_test_split

from ultralytics import YOLO


# Generate COCO dataset from YOLO model
def export_coco_dataset_from_prediction(data_path, folder_name, model_name="yolov8n.pt"):
    model_path = os.path.join(os.path.dirname(os.getcwd()), 'models')
    print(model_path)
    model = YOLO(os.path.join(model_path, model_name))

    images_path = os.path.join(data_path, 'images' + '/' + folder_name)

    dest_path = os.path.join(data_path, 'coco_datasets' + '/' + folder_name)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    dest_images_path = os.path.join(dest_path, 'images')
    if not os.path.exists(dest_images_path):
        os.mkdir(dest_images_path)

    dest_coco_path = os.path.join(dest_path, 'annotations')
    if not os.path.exists(dest_coco_path):
        os.mkdir(dest_coco_path)

    
    coco_dict ={}
    frame_id = 1
    obj_id = 1
    categories_dict = [{"id" : 1, "name" : "person", "supercategory" : None},
                    {"id" : 2, "name" : "ball", "supercategory" : None}]

    creation_date = date.today()
    year = creation_date.year
    coco_info = {
        'contributor': 'Khoa Nguyen, Huy Nguyen',
        'description': folder_name,
        'url': '',
        'version': 0,
        'date_created': str(creation_date),
        'year': year,
    }

    images = []
    annotations = []


    for image_name in os.listdir(images_path):
        frame = os.path.join(images_path, image_name)
        # Get image shape for COCO dataset:
        h, w, _ = cv2.imread(frame, cv2.IMREAD_UNCHANGED).shape

        # Set image metadata for COCO dataset:
        metadata = {"height" : h,
                    "width" : w,
                    "id" : frame_id,
                    "file_name": image_name}
        images.append(metadata)

        # Run YOLOv8 inference on the frame
        results = model(frame, classes=[0, 32])

        # Copy original image to coco dataset destination
        shutil.copy(frame, os.path.join(dest_images_path, image_name))

        # Write information to coco dataset:
        boxes = results[0].boxes
        for box in boxes:
            (startX, startY, endX, endY) = box.xyxy.cpu().detach().int().tolist()[0]
            class_id = box.cls[0].item()
            bbox = {"id" : obj_id,
                    "image_id" : frame_id,
                    "category_id" : 1 if class_id == 0.0 else 2,
                    "segmentation" : [],
                    "bbox" : [startX, startY, endX-startX, endY-startY],
                        "area" : (endX-startX) * (endY-startY),
                    "iscrowd" : 0}
            
            annotations.append(bbox)
            obj_id += 1
        frame_id += 1


    coco_dict["info"] = coco_info
    coco_dict["license"] = {'name': folder_name,
                            'id': '',
                            'url': ''}
    coco_dict["categories"] = categories_dict
    coco_dict["images"] = images
    coco_dict["annotations"] = annotations

    with open(os.path.join(dest_coco_path, 'instances_default.json'), 'w') as coco:
        json.dump(coco_dict, coco)


# Merge two COCO dataset
def merge_coco_dataset(coco_dataset_path_1, coco_dataset_path_2, dest_path=None):
    coco_dataset_path_1 = os.path.abspath(coco_dataset_path_1)
    coco_dataset_path_2 = os.path.abspath(coco_dataset_path_2)

    # Load data
    images_path_1 = os.path.join(coco_dataset_path_1, 'images')
    images_path_2 = os.path.join(coco_dataset_path_2, 'images')
    coco_path_1 = os.path.join(coco_dataset_path_1, 'annotations/instances_default.json')
    coco_path_2 = os.path.join(coco_dataset_path_2, 'annotations/instances_default.json')

    coco_1 = json.load(open(coco_path_1))
    coco_2 = json.load(open(coco_path_2))

    # Create destination path
    if dest_path is None or not os.path.isdir(dest_path):
        dest_path = os.path.join(os.path.dirname(coco_dataset_path_1), 
                                 (os.path.basename(coco_dataset_path_1) + '_' + os.path.basename(coco_dataset_path_2)))
    else:
        dest_path = os.path.join(dest_path, 
                                 (os.path.basename(coco_dataset_path_1) + '_' + os.path.basename(coco_dataset_path_2)))
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    # Merged coco destination path
    dest_coco_path = os.path.join(dest_path, 'annotations')
    if not os.path.exists(dest_coco_path):
        os.mkdir(dest_coco_path)
    
    # Merged images folder
    dest_images_path = os.path.join(dest_path, 'images')
    if not os.path.exists(dest_images_path):
        os.mkdir(dest_images_path)

    # Update metadata
    final_coco = coco_1
    final_coco["info"]["description"] = "merge_files"
    final_coco["license"]["name"] = "merge_files"

    # Update image_name for second coco and move images from two dataset into destination folder
    image_list_1 = coco_1["images"]
    image_list_2 = coco_2["images"]
    image_name_list_1 = [i["file_name"] for i in image_list_1]
    for image_name_1 in image_name_list_1:
        image_path_1 = os.path.join(images_path_1, image_name_1)
        shutil.copy(image_path_1, os.path.join(dest_images_path, image_name_1))

    for i in range(len(image_list_2)):
        image_list_2[i]["id"] = image_list_2[i]["id"] + len(image_list_1)
        old_name = image_list_2[i]["file_name"]
        # Change name of image in the 2nd coco json in case of duplication
        if image_list_2[i]["file_name"] in image_name_list_1:
            new_name = image_list_2[i]["file_name"].replace(".png","_dup.png")
            image_list_2[i]["file_name"] = new_name
        else:
            new_name = image_list_2[i]["file_name"]

        # Move images from 2nd coco dataset to new destination
        image_path_2 = os.path.join(images_path_2, old_name)
        shutil.copy(image_path_2, os.path.join(dest_images_path, new_name))
    final_coco["images"] = sorted(image_list_1 + image_list_2, key=lambda x: x["id"])

    # Update annotations 
    annotation_list_1 = coco_1["annotations"]
    annotation_list_2 = coco_2["annotations"]
    for i in range(len(annotation_list_2)):
        annotation_list_2[i]["image_id"] = annotation_list_2[i]["image_id"] + len(image_list_1)
        annotation_list_2[i]["id"] = annotation_list_2[i]["id"] + len(annotation_list_1) 
        
    final_coco["annotations"] = sorted(annotation_list_1 + annotation_list_2, key=lambda x: x["id"])


    with open(os.path.join(dest_coco_path, 'instances_default.json'), 'w') as f:
        json.dump(final_coco, f)
            

# Convert COCO dataset to YOLO dataset format        
def coco2yolo(train_dataset_path, val_dataset_path=None, dest_path=None, split_val=True):

    def ann2txt(coco_file, image_info_list, images_path, dest_images_path, dest_labels_path):
        for image_info in image_info_list:
            image_name = image_info['file_name']
            image_w, image_h = image_info['width'], image_info['height']

            with open(os.path.join(dest_labels_path, image_name[:-4]+'.txt'), 'w') as f:
                for annotation in coco_file['annotations']:
                    if annotation['image_id'] != image_info['id']:
                        continue
                    
                    # Convert annotation information to YOLO format
                    label_id = annotation['category_id']
                    startX, startY, w, h = annotation['bbox']
                    endX, endY = [startX + w, startY + h]

                    x_center=((2*startX+w)/(2*image_w))
                    y_center=((2*startY+h)/(2*image_h))
                    w_box=w/image_w
                    h_box=h/image_h

                    line = '{} {} {} {} {}'.format(label_id-1, x_center, y_center, w_box, h_box)
                    f.write(line + '\n')

            # Resize images to HD scale
            image = cv2.imread(os.path.join(images_path, image_name), cv2.IMREAD_UNCHANGED)
            resized_image = cv2.resize(image, (1280,720))
            cv2.imwrite(os.path.join(dest_images_path, image_name), resized_image)
        
        return dest_images_path

    if dest_path is None:
        dest_path = os.path.join(os.path.dirname(train_dataset_path), os.path.basename(train_dataset_path) + '_yolov8')
    
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    train_dest_path = os.path.join(dest_path, 'train')
    test_dest_path = os.path.join(dest_path, 'test/images')
    val_dest_path = os.path.join(dest_path, 'valid')
    train_dest_images_path = os.path.join(train_dest_path, 'images')
    train_dest_labels_path = os.path.join(train_dest_path, 'labels')
    val_dest_images_path = os.path.join(val_dest_path, 'images')
    val_dest_labels_path = os.path.join(val_dest_path, 'labels')
    for dest_paths in [train_dest_path, val_dest_path, train_dest_images_path, train_dest_labels_path, val_dest_images_path, val_dest_labels_path]:
        if not os.path.exists(dest_paths):
            os.mkdir(dest_paths)

    dest_images_path_list = [train_dest_images_path, val_dest_images_path]
    dest_labels_path_list = [train_dest_labels_path, val_dest_labels_path]

    if val_dataset_path is not None:
        split_val = False
        
        for i in range(2):
            dataset_path = [train_dataset_path, val_dataset_path][i]
            coco_path = os.path.join(dataset_path, 'annotations/instances_default.json')
            images_path = os.path.join(dataset_path, 'images')
            coco = json.load(open(coco_path))

            image_info_list = coco['images']

            dest_images_path = dest_images_path_list[i]
            dest_labels_path = dest_labels_path_list[i]

            ann2txt(coco, image_info_list, images_path, dest_images_path, dest_labels_path)

    
    elif val_dataset_path is None and split_val:
        coco_path = os.path.join(train_dataset_path, 'annotations/instances_default.json')
        images_path = os.path.join(train_dataset_path, 'images')
        coco = json.load(open(coco_path))

        image_list = coco['images']

        train_image_list, val_image_list = train_test_split(image_list, shuffle=True, test_size=0.2, random_state=42)

        for i in range(2):
            dest_images_path = dest_images_path_list[i]
            dest_labels_path = dest_labels_path_list[i]

            image_info_list = [train_image_list, val_image_list][i]

            ann2txt(coco, image_info_list, images_path, dest_images_path, dest_labels_path)


    else:
        return "Validation set path required or set split_val to True to split train dataset to train and validation set"
    
    yaml_path = os.path.join(dest_path, 'data.yaml')

    info_dict = {'train': train_dest_images_path,
                'val': val_dest_images_path,
                'test' : test_dest_path, 
                'nc': len(coco['categories']),
                'names': [category['name'] for category in coco['categories']]
                }
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(info_dict, yaml_file)