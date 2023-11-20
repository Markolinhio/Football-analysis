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
import albumentations
from tqdm import tqdm

from sklearn.cluster import KMeans
from collections import Counter
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from pathlib import Path
from yaml.loader import SafeLoader
from albumentations.core.transforms_interface import ImageOnlyTransform
from datetime import date
plt.rcParams["figure.figsize"] = (24,18)

from sklearn.model_selection import train_test_split

from ultralytics import YOLO


# Detect largest contour in the image along with its bounding rectangle and convex hull. Used for field segmentation and misc task when needed
def detect_largest_contour(image, threshold=False):
    # Contour detection
    if len(image.shape) == 2 or image.shape[2] == 1:
        if threshold:
            _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)
            
        else:
            contours, _ = cv2.findContours(image=image, mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)

    else:
        return "Image should be one channel"
        
    # Take the largest contour
    contours = max(contours, key=cv2.contourArea)

    # Find the bounding box corresponding to the contour
    rect = np.int16(cv2.boundingRect(contours))

    # Find the convex hull corresponding to the bounding box
    convex_hull = cv2.convexHull(contours)

    return contours, rect, convex_hull



# Generate COCO dataset from YOLO model
def export_coco_dataset_from_prediction(data_path, folder_name, model_name="yolov8n.pt"):
    # Declare input and output paths
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

    # Initialize fixed COCO attributes
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

    # Predict and put the information to COCO dict
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
                    "area" : int((endX-startX) * (endY-startY)),
                    "iscrowd" : 0}
            
            annotations.append(bbox)
            obj_id += 1
        frame_id += 1

    # Write COCO dict
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
def coco2yolo(train_dataset_path, val_dataset_path=None, test_dataset_path=None, dest_path=None, split_val=True):

    def ann2txt(coco_file, image_info_list, images_path, dest_images_path, dest_labels_path):
        for image_info in tqdm(image_info_list):
            image_name = image_info['file_name']
            image_w, image_h = image_info['width'], image_info['height']

            # Write COCO bounding box as YOLO information: 
            # COCO: starting coodinates and width, height of the bounding box, category_id starts at 1
            # YOLO: to center coordinate of the bounding box and its width and height, category_id starts at 0 (COCO id - 1)
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

    # Declare destination paths
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

    # If validation set path declared, write images and label to the corresponding destination
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

    # If no validation set path is declared, split the images and label list as 80/20 and write images and label to the corresponding destination
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

    # If no validation set declared and split_val as False, return error
    else:
        return "Validation set path required or set split_val to True to split train dataset to train and validation set"
    
    if test_dataset_path is not None:
        # Create corresponding destination folders if test_dataset_path is available

        test_dest_path = os.path.join(dest_path, 'test')
        test_dest_images_path = os.path.join(test_dest_path, 'images')
        test_dest_labels_path = os.path.join(test_dest_path, 'labels')

        for dest_paths in [test_dest_path, test_dest_images_path, test_dest_labels_path]:
            if not os.path.exists(dest_paths):
                os.mkdir(dest_paths)
                
        dataset_path = test_dataset_path
        coco_path = os.path.join(dataset_path, 'annotations/instances_default.json')
        images_path = os.path.join(dataset_path, 'images')
        coco = json.load(open(coco_path))

        image_info_list = coco['images']

        ann2txt(coco, image_info_list, images_path, test_dest_images_path, test_dest_labels_path)

    # Write YAML metada of YOLO dataset
    yaml_path = os.path.join(dest_path, 'data.yaml')

    info_dict = {'train': os.path.relpath(train_dest_images_path, dest_path),
                'val': os.path.relpath(val_dest_images_path, dest_path),
                'test' : os.path.relpath(test_dest_path, dest_path), 
                'nc': len(coco['categories']),
                'names': [category['name'] for category in coco['categories']]
                }
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(info_dict, yaml_file)
            

def augment_yolo(yaml_path, dataset_path):
    def salt_pepper(image):
        salt_vs_pepper = 0.5                  
        amount = 0.01
        out = np.copy(image)
        
        # Salt mode
        num_salt = np.ceil(amount * image.size * salt_vs_pepper)
        coords = np.array([np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape])
        salt_idx = np.where(np.array(coords[2] == 1))
        for idx in salt_idx:
            h, w = coords[0][idx], coords[1][idx]
            out[h, w, :] = 255
        #out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - salt_vs_pepper))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        pepper_idx = np.where(np.array(coords[2] == 1))
        for idx in pepper_idx:
            h, w = coords[0][idx], coords[1][idx]
            out[h, w, :] = 0
        return out 

    class Salt_pepper(ImageOnlyTransform):
        def apply(self, img, **params):
            return salt_pepper(img)

    with open(yaml_path) as f:
        yaml_file = yaml.load(f, Loader=SafeLoader)

    # Get file path for augmentation
    train_path = yaml_file["train"]
    
    # Augmentation pipeline of three augmentation:
    # Hue saturation transform between -25% and 25%
    # Random Brightness between -10% and 10%
    # Salt and pepper of 1% of pixels
    transform = albumentations.Compose([
        albumentations.HueSaturationValue(p=0.5), #value dau???
        albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5), #sao lai co contrast limit? @@
        Salt_pepper()])

    # Iterate over the images in the dataset path, augment them, and write them to the corresponding destination
    file_path = os.path.join(dataset_path, train_path)
    print(file_path)

    images_list = os.listdir(file_path)

    for image_name in tqdm(images_list):
        image_path = os.path.join(file_path, image_name)
        label_path = os.path.join(os.path.dirname(file_path)+'/labels', image_name[:-4]+'.txt')
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

        # Run an image through augmentation pipeline of 2^3 times to try getting all combination of augmentation
        for i in range(8):
            transformed_image = transform(image=image)['image']
            transformed_image_name = image_name[:-4] + '_transformed_' + str(i) + '.png'
            transformed_label_name = image_name[:-4] + '_transformed_' + str(i) + '.txt'
            cv2.imwrite(os.path.join(file_path,transformed_image_name), transformed_image)
            shutil.copy(label_path, os.path.join(os.path.dirname(file_path)+'/labels', transformed_label_name))

    print(len(os.listdir(os.path.join(dataset_path, train_path))), len(os.listdir(os.path.dirname(file_path)+'/labels')))


# Replace old def "asscalar" to .item()
def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)


# Reduction of bounding box area so that it captures the player shirts only
def reduce_area(startX, endX, startY, endY, threshold = 0.4):
    # Input the coordinates of a bounding boxes and output the desired box
    # Threshold is % of the remaining area
    # end_Y is strictly set to half od the height of the image so that the code ignores the pants
    # The correct version of the code with Y axis
    # new_endY = round(endY - (1-np.sqrt(z))*(endY-startY)/2) 

    new_startY = round(startY + (1-np.sqrt(threshold))*(endY - startY)/2)
    new_endY   = round((startY + endY)/2)                   
    new_startX = round(startX + (1-np.sqrt(threshold))*(endX - startX)/2)
    new_endX   = round(endX - (1-np.sqrt(threshold))*(endX - startX)/2)

    return new_startX, new_endX, new_startY, new_endY


def box_to_features(rgb_image,boxes, frame_number):

    # Input: rgb_images and the resulting bounding boxes from the model
    # Output: Lists of extracted features namely 
    box_coord_list  = []  # original coordinates of each box, used for testing purposes
    player_center_coord = []  # coordinate of the center of each box
    assignment     = []  # main color pigment of each box     
    bbox_annotation = []

    #Compute the hue of grass:
    grass_average = rgb_image[:,:,::-1].mean(axis=0).mean(axis=0)
    obj_number = 0
    # For each box, extract features
    for box in boxes:
        crop = box.xyxy
        (x_1, y_1, x_2, y_2) = np.concatenate(crop.cpu().detach().int().tolist()) # Orgin coordinates

        box_coord_list.append(((x_1, y_1), (x_2, y_2)))       # save original coordinates
        player_center_coord.append((0.5*(x_1 + x_2), 0.5*(y_1 + y_2)))             # Coordinate of the center of the box
        class_id = box.cls[0].item()
        bbox = {"id" : obj_number,
                "image_id" : frame_number,
                "category_id" : 7 if class_id == 0.0 else 0,
                "segmentation" : [],
                "bbox" : [x_1, y_1, x_2-x_1, y_2-y_1],
                "area" : (x_2-x_1) * (y_2-y_1),
                "iscrowd" : 0}
        bbox_annotation.append(bbox)
        # If box class is a person then proceed
        if box.cls[0].item() == 0:   
            # For the main color, we first reduce area of search to only shirts
            # Further filter to remove all the grass hue from the image
            # With all the accepted pixels, we compute the average color of the shirts
            (new_x_1,new_x_2,new_y_1,new_y_2) = reduce_area(x_1,x_2,y_1,y_2) 
            cropped = rgb_image[new_y_1:new_y_2, new_x_1:new_x_2]
            test_crop = cropped[:,:,::-1].copy()
            accepted_pixel = []
            for row_idx in range(cropped.shape[0]):
                for col_idx in range(cropped.shape[1]):        
                    current_color = test_crop[row_idx][col_idx]
                    if sum(abs(current_color - grass_average)) > 60:
                        accepted_pixel.append(current_color)
            final_color = np.mean(accepted_pixel,axis = 0)
            assignment.append(final_color)                     # assignment by color
        
        obj_number += 1 

    return assignment, player_center_coord, box_coord_list, bbox_annotation


# Remove the audiences from the image
def pitch_segment(rgb_image, visualize=False):
    orig_h, orig_w, _ = rgb_image.shape

    ## Filter green pixels
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    hue = hsv_image[:, :, 0]
    lower_green = 35
    upper_green = 100

    mask_green = cv2.inRange(hue, lower_green, upper_green)
    green_filtered_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_green)

    _, rect, _ = detect_largest_contour(mask_green)

    min_x, min_y, w, h = rect
    max_x, max_y = [min_x + w, min_y + h]

    # Crop the image
    field_cropped_image = cv2.resize(rgb_image[min_y:max_y, min_x:max_x], (orig_w, orig_h)) # Crop and Resize to original size

    if visualize:
        plt.figure()
        plt.imshow(mask_green)
        temp = rgb_image.copy()
        cv2.rectangle(temp, (min_x, min_y), (max_x, max_y), (255, 0, 0), 3)
        plt.figure()
        plt.imshow(temp)

    return field_cropped_image


# Convert rgb to CLE_lab for computing the difference between 2 different colors
def rgb_to_lab_color(rgb_color):
    # Input rgb color
    # Output CLE_lab color
    srgb = sRGBColor(rgb_color[0],rgb_color[1],rgb_color[2])
    lab = convert_color(srgb, LabColor)

    return lab


# Check if there is a mismatch color in the same team and change the labels accordingly
def misc_in_teams(labels, assignment, teams, threshold = 50):
    # Input: labels, their color values, and counter for the labels
    # Output: Updated labels

    # Analyze the results from KMeans to get 2 teams
    team_1 = np.where(labels == teams[0][0])[0]
    team_2 = np.where(labels == teams[1][0])[0]

    # Compute the average color of each team with the index
    team_1_color = np.mean([assignment[i] for i in team_1],axis =0)
    team_2_color = np.mean([assignment[i] for i in team_2],axis =0)
        
    # Convert rgb to lab
    lab_team_1_color = rgb_to_lab_color(team_1_color)
    lab_team_2_color = rgb_to_lab_color(team_2_color)
    lab_assignment = [rgb_to_lab_color(x) for x in assignment]

    # All the number labels the KMeans gives
    updated_labels = labels
    team_1_label = teams[0][0]
    team_2_label = teams[1][0]
    misc_label   = teams[2][0]

    # Compute the difference of colors within their own team
    team_1_check = [delta_e_cie2000(lab_team_1_color, lab_assignment[i]) for i in team_1]
    team_2_check = [delta_e_cie2000(lab_team_2_color, lab_assignment[i]) for i in team_2]
    
    # Check if there is a mismatch color within 2 teams
    convert_misc_1_idx = [i for i in range(len(team_1_check)) if team_1_check[i] > threshold] 
    convert_misc_2_idx = [i for i in range(len(team_2_check)) if team_2_check[i] > threshold]
    
    # If yes, then change the labels accordingly
    if len(convert_misc_1_idx) > 0:
            for x in convert_misc_1_idx:
                if delta_e_cie2000(lab_team_2_color,lab_assignment[x]) < 30:
                    updated_labels[team_1[x]] = team_2_label
                else:
                    updated_labels[team_1[x]] = misc_label
            
    if len(convert_misc_2_idx) > 0:
            for y in convert_misc_2_idx:
                if delta_e_cie2000(lab_team_1_color,lab_assignment[y]) < 30:
                    updated_labels[team_2[y]] = team_1_label
                else:
                    updated_labels[team_2[y]] = misc_label
    
    return updated_labels, lab_team_1_color, lab_team_2_color


# Identify the position of the misc label
def position_text(player_center_coord,new_team_1,new_team_2,new_misc):
    # Input: Players x coordinates and the index of the all classes
    # Output: determine the position of the misc label (Left or Right)
    player_x_coord = np.array(player_center_coord)[:, 0]

    misc_pos = [player_x_coord[i] for i in new_misc]
    avg_x_team_1 = np.mean([player_x_coord[i] for i in new_team_1])
    avg_x_team_2 = np.mean([player_x_coord[i] for i in new_team_2])

    # Identify left vs right position. The metric can be changed if needed
    if avg_x_team_1 < avg_x_team_2:
        left_coord = avg_x_team_1
        left_team  = "_Team_1"
        right_team = "_Team_2"
    else:
        left_coord = avg_x_team_2
        left_team  = "_Team_2"
        right_team = "_Team_1"
    
    # Update text
    for i in range(len(new_misc)):
        if misc_pos[i] <= left_coord:
            misc_pos[i] = "Left"
        else:
            misc_pos[i] = "Right"

    return misc_pos, left_team, right_team


def update_misc_color(team_1_idx, team_2_idx, misc_idx, global_color_dict, current_color_dict, labels):
    global_misc_lab_color = [rgb_to_lab_color(x) for x in global_color_dict['misc_color']]
    team_1_diff = [delta_e_cie2000(global_color_dict['team_1_color'], x) for x in current_color_dict['current_misc_lab_colors']]
    team_2_diff = [delta_e_cie2000(global_color_dict['team_2_color'], x) for x in current_color_dict['current_misc_lab_colors']]
    current_misc_position, left_team, right_team = position_text(current_color_dict['player_center_coord_list'], team_1_idx, team_2_idx, misc_idx)
    switch = [False]*len(current_color_dict['current_misc_lab_colors'])

    # Check availability of the misc clr:
    if len(global_color_dict['misc_color']) == 0:
        for j in range(len(current_color_dict['current_misc_colors'])):
            # If the same shade of color then update the param of existing global, but if it is not then addd into the list
            if (team_1_diff[j] < 40):
                current_color_dict["label"][j] = "Team 1"
                labels[misc_idx[j]] = "Team_1"
                switch[j] = True
            elif (team_2_diff[j] < 40):
                current_color_dict["label"][j] = "Team 2"
                labels[misc_idx[j]] = "Team_2"
                switch[j] = True
            else:
                if switch[j] == False:
                    global_color_dict['misc_box_coord'].append(current_color_dict['current_misc_box_coord'][j])
                    global_color_dict['misc_color'].append(current_color_dict['current_misc_colors'][j])
                    global_color_dict['misc_position'].append(current_misc_position[j])
                    global_color_dict['misc_frequency'].append(1)
    else:
        #global vs local:
        for j in range(len(current_color_dict['current_misc_lab_colors'])):
            # If the same shade of color then update the param of existing global, but if it is not then add into the list
            if (team_1_diff[j] < 40):
                current_color_dict["label"][j] = "Team 1"
                labels[misc_idx[j]] = "Team_1"
                switch[j] = True
            elif (team_2_diff[j] < 40):
                current_color_dict["label"][j] = "Team 2"
                labels[misc_idx[j]] = "Team_2"
                switch[j] = True
            else:
                for i in range(len(global_color_dict['misc_color'])):
                    misc_diff = delta_e_cie2000(global_misc_lab_color[i], 
                                                current_color_dict['current_misc_lab_colors'][j])
                    if misc_diff < 40:
                        if current_misc_position[j] == global_color_dict['misc_position'][i]:
                            global_color_dict['misc_frequency'][i] += 1
                        else:
                            global_color_dict['misc_frequency'][i] += 1
                            global_color_dict['misc_position'][i] = "Mid"
                        
                        # Update the labels if there is a match between global vs local color
                        if global_color_dict['misc_frequency'][i] >= 3:
                            if global_color_dict['misc_position'][i] == "Mid":
                                current_color_dict["label"][j] = "Referee"
                                labels[misc_idx[j]] = "Referee"
                            elif global_color_dict['misc_position'][i] == "Left":
                                current_color_dict["label"][j] = "Keeper" # Exact team update is later
                                labels[misc_idx[j]] = "Keeper" + left_team
                            elif global_color_dict['misc_position'][i] == "Right":
                                current_color_dict["label"][j] = "Keeper" # Exact team update is later
                                labels[misc_idx[j]] = "Keeper" + right_team
                        switch[j] = True

                    

        for j in range(len(current_color_dict['current_misc_lab_colors'])):
            if switch[j] == False:
                if len(global_color_dict['misc_color']) <= 4:
                    # Update global dict
                    global_color_dict['misc_box_coord'].append(current_color_dict['current_misc_box_coord'][j])
                    global_color_dict['misc_color'].append(current_color_dict['current_misc_colors'][j])
                    global_color_dict['misc_position'].append(current_misc_position[j])
                    global_color_dict['misc_frequency'].append(1)
                else:
                    if min(global_color_dict['misc_frequency']) == 1:
                        k = global_color_dict['misc_frequency'].index(min(global_color_dict['misc_frequency']))
                        global_color_dict['misc_box_coord'][k] = current_color_dict['current_misc_box_coord'][j]
                        global_color_dict['misc_color'][k] = current_color_dict['current_misc_colors'][j]
                        global_color_dict['misc_position'][k] = current_misc_position[j]
                        global_color_dict['misc_frequency'] = 1

    return global_color_dict

def convert_lab_to_rgb(lab_color):
        rgb = convert_color(lab_color, sRGBColor)
        return (rgb.rgb_r,rgb.rgb_g,rgb.rgb_b)


def visualize_with_labels(rgb_image, team_1_idx, team_2_idx, misc_idx, labels, global_color_dict, current_misc_dict, box_coord_list):
    final_image = rgb_image.copy()
    (r_1,g_1,b_1) = convert_lab_to_rgb(global_color_dict["team_1_color"])
    (r_2,g_2,b_2) = convert_lab_to_rgb(global_color_dict["team_2_color"])

    for i in team_1_idx:
        (x_1, y_1), (x_2, y_2) = box_coord_list[i]
        final_image = cv2.rectangle(final_image, (x_1, y_1), (x_2, y_2), (b_1,g_1,r_1))  # Fix color later
        final_image = cv2.putText(img = final_image, text= labels[i], org=(x_1, y_1), color = (b_1,g_1,r_1),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1.0,thickness = 1)
    
    for i in team_2_idx:
        (x_1, y_1), (x_2, y_2) = box_coord_list[i]
        final_image = cv2.rectangle(final_image, (x_1, y_1), (x_2, y_2), (b_2,g_2,r_2))  # Fix color later
        final_image = cv2.putText(img = final_image, text= labels[i], org=(x_1, y_1), color = (b_2,g_2,r_2),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1.0,thickness = 1)

    for i in range(len(misc_idx)):
        (x_1, y_1), (x_2, y_2) = current_misc_dict["current_misc_box_coord"][i]
        (r_3, g_3, b_3) = convert_lab_to_rgb(current_misc_dict["current_misc_lab_colors"][i])
        final_image = cv2.rectangle(final_image, (x_1, y_1), (x_2, y_2), (b_3,g_3,r_3))  # Fix color later
        final_image = cv2.putText(img = final_image, text=current_misc_dict["label"][i], org=(x_1, y_1), color = (b_3,g_3,r_3),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1.0,thickness = 1)

    plt.figure()
    plt.imshow(final_image)


def update_bbox_label(bbox_annotation,labels):
    updated_bbox_annotation = bbox_annotation.copy()
    i = 0
    for box in updated_bbox_annotation:
        if box["category_id"] != 0:
            if labels[i] == "Team 1":
                box["category_id"] = 1
            elif labels[i] == "Team 2":
                box["category_id"] = 2
            elif labels[i] == "Referee":
                box["category_id"] = 5
            elif labels[i] == "Keeper Team 1":
                box["category_id"] = 3
            elif labels[i] == "Keeper Team 2":
                box["category_id"] = 4
            elif labels[i] == "Misc":
                box["category_id"] = 6
            i += 1
    return updated_bbox_annotation


def write_coco_with_player_differentiation(vid_name, model_name, data_path, model_path):
    model = YOLO(os.path.join(model_path, model_name))
    images_path = os.path.join(data_path, 'images' + '/' + vid_name)
    destination_path = os.path.join(data_path, 'coco_datasets/' + vid_name)
    destination_path_coco = os.path.join(destination_path, 'annotations')
    destination_path_image = os.path.join(destination_path, 'images')
    for destination in [destination_path, destination_path_coco, destination_path_image]:
        if not os.path.exists(destination):
            os.mkdir(destination)

    # Initialize fixed COCO attributes
    coco_dict ={}
    frame_id = 1
    obj_id = 1
    categories_dict = [{"id" : 0, "name" : "ball", "supercategory" : None},
                    {"id" : 1, "name" : "player_team_1", "supercategory" : None},
                    {"id" : 2, "name" : "player_team_2", "supercategory" : None},
                    {"id" : 3, "name" : "keeper_team_1", "supercategory" : None},
                    {"id" : 4, "name" : "keeper_team_2", "supercategory" : None},
                    {"id" : 5, "name" : "referee", "supercategory" : None},
                    {"id" : 6, "name" : "misc", "supercategory" : None},
                    {"id" : 7, "name" : "player", "supercategory" : None}]


    creation_date = date.today()
    year = creation_date.year
    coco_info = {
        'contributor': 'Khoa Nguyen, Huy Nguyen',
        'description': vid_name,
        'url': '',
        'version': 0,
        'date_created': str(creation_date),
        'year': year,
    }

    # Attributes to identify colors
    global_color_dict = {'team_1_color' : [],
                        'team_1_position' : [],
                        'team_2_color' : [],
                        'team_2_position' : [],
                        'misc_color' : [],
                        'misc_position' : [],
                        'misc_frequency' : [],
                        'misc_box_coord' : []}


    # Saved info to be write COCO file:
    all_image_info = []
    all_bbox_info = []

    frame_num = 0

    #first_batch = os.listdir(images_path)[0:20]
    #mage_batch = [os.path.join(images_path, x) for x in first_batch]

    for image_name in os.listdir(images_path):
        
        # Crop out the audiences via pitch segmentation
        image_ad  = os.path.join(images_path, image_name)
        rgb_image = cv2.imread(image_ad, cv2.COLOR_BGR2RGB)
        rgb_image = pitch_segment(rgb_image)
        (h, w, _) = rgb_image.shape


        # Set image metadata for COCO dataset:
        metadata = {"height" : int(h),
                    "width" : int(w),
                    "id" : frame_num,
                    "file_name": image_name}     # Work on image names later
        all_image_info.append(metadata)

        # Run yolov8n on the current image and extract features from resulting boxes
        results = model.predict(rgb_image)
        boxes = results[0].boxes
        assignment, player_center_coord_list, box_coord_list, bbox_annotation = box_to_features(rgb_image, boxes, frame_num)

        # Applies KNN with 3 clusters to find the most prominent colors as in rgb_color
        try:
            kmeans = KMeans(n_clusters=3)
            s=kmeans.fit(assignment)
            labels=kmeans.labels_

            # Extract the labels into 2 teams and misc color
            teams = Counter(labels).most_common(3)
            team_1_label = teams[0][0]
            team_2_label = teams[1][0]
            misc_label   = teams[2][0]

            #Return correct labels of (numbers) and the 2 team lab colors
            labels, lab_team_1_color, lab_team_2_color  = misc_in_teams(labels,assignment,teams)

            # Labeling via strings and consistency checks so that team X is always color Y:
            if frame_num == 0:
                global_color_dict["team_1_color"] = lab_team_1_color 
                global_color_dict["team_2_color"] = lab_team_2_color

                labels = list(map(lambda x: x if x != team_1_label else 'Team 1', labels))
                labels = list(map(lambda x: x if x != team_2_label else 'Team 2', labels))
                labels = list(map(lambda x: x if x != misc_label else 'Misc', labels))
                
            if frame_num > 0:
                global_1_vs_local_1 = delta_e_cie2000(global_color_dict["team_1_color"], lab_team_1_color)
                global_1_vs_local_2 = delta_e_cie2000(global_color_dict["team_1_color"], lab_team_2_color)
                if  global_1_vs_local_1 < global_1_vs_local_2:
                    #print("Team 1 global is match with team 1 local")
                    labels = list(map(lambda x: x if x != team_1_label else 'Team 1', labels))
                    labels = list(map(lambda x: x if x != team_2_label else 'Team 2', labels))
                else: 
                    #print("Team 1 global is not match with team 1 local")
                    labels = list(map(lambda x: x if x != team_2_label else 'Team 1', labels))
                    labels = list(map(lambda x: x if x != team_1_label else 'Team 2', labels))
                labels = list(map(lambda x: x if x != misc_label else 'Misc', labels)) 

            # Updated teams indices
            team_1_idx = np.where(np.array(labels) == "Team 1")[0]
            team_2_idx = np.where(np.array(labels) == "Team 2")[0]
            misc_idx   = np.where(np.array(labels) == "Misc")[0] 
            

            # Update the misc color: 
            current_misc_box_coord = [box_coord_list[i] for i in misc_idx]
            current_misc_color = [assignment[i] for i in misc_idx]
            current_misc_lab_colors   = [rgb_to_lab_color(x) for x in current_misc_color]

            current_misc_dict = {'player_center_coord_list': player_center_coord_list,
                                'current_misc_box_coord': current_misc_box_coord,
                                'current_misc_colors': current_misc_color,
                                'current_misc_lab_colors': current_misc_lab_colors,
                                'label': ["Misc"]*len(current_misc_color)}

            
            global_color_dict = update_misc_color(team_1_idx, team_2_idx, misc_idx, 
                                                global_color_dict, current_misc_dict, labels)
            

            #visualize_with_labels(rgb_image, team_1_idx, team_2_idx, misc_idx, labels, global_color_dict, current_misc_dict, box_coord_list)
            final_bbox = update_bbox_label(bbox_annotation,labels)
            for box in final_bbox:
                all_bbox_info.append(box)

            frame_num +=1

        except:
            for box in final_bbox:
                all_bbox_info.append(bbox_annotation)
            frame_num +=1
        
    # Write COCO dict
    coco_dict["info"] = coco_info
    coco_dict["license"] = {'name': vid_name,
                                'id': '',
                                'url': ''}
    coco_dict["categories"] = categories_dict
    coco_dict["images"] = all_image_info
    coco_dict["annotations"] = all_bbox_info
    
    # Custom json encoder
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    
    # Write json file
    with open(os.path.join(destination_path_coco, 'instances_default.json'), 'w') as coco:
        json.dump(coco_dict, coco, cls=NpEncoder)

    print("Done")
