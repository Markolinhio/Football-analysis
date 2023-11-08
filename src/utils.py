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
                        "area" : (endX-startX) * (endY-startY),
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
def coco2yolo(train_dataset_path, val_dataset_path=None, dest_path=None, split_val=True):

    def ann2txt(coco_file, image_info_list, images_path, dest_images_path, dest_labels_path):
        for image_info in image_info_list:
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

    # Declare destination path
    test_path = yaml_file["test"]
    train_path = yaml_file["train"]
    val_path = yaml_file["val"]
    storage = [test_path,train_path,val_path]
    
    # Augmentation pipeline of three augmentation:
    # Hue saturation transform between -25% and 25%
    # Random Brightness between -10% and 10%
    # Salt and pepper of 1% of pixels
    transform = albumentations.Compose([
        albumentations.HueSaturationValue(p=0.5), #value dau???
        albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5), #sao lai co contrast limit? @@
        Salt_pepper()])

    # Iterate over the images in the dataset path, augment them, and write them to the corresponding destination
    for file_path in storage:
        file_path = os.path.join(dataset_path, file_path)
        if not os.path.exists(file_path):
            continue
        images_list = os.listdir(file_path)

        generate_data = True # Biet cai generate data lam gi ko ma ghi o day? @@
        for image_name in images_list:
            image_path = os.path.join(file_path, image_name)
            label_path = os.path.join(os.path.dirname(file_path)+'/labels', image_name[:-4]+'.txt')
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

            # Run an image through augmentation pipeline of 2^3 times to try getting all combination of augmentation
            for i in range(8):
                transformed_image = transform(image=image)['image']
                transformed_image_name = image_name[:-4] + '_transformed_' + str(i) + '.png'
                transformed_label_name = image_name[:-4] + '_transformed_' + str(i) + '.txt'
                if generate_data: # Biet cai generate data lam gi ko ma ghi o day? @@
                    cv2.imwrite(os.path.join(file_path,transformed_image_name), transformed_image)
                    shutil.copy(label_path, os.path.join(os.path.dirname(file_path)+'/labels', transformed_label_name))

        print(len(os.listdir(image_path)), len(os.listdir(label_path)))
    
# Replace old def "asscalar" to .item()
def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)

# Reduction of bounding box area so that it captures the player shirts only
def area_reduction(startX, endX, startY, endY, threshold = 0.4):
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

def box_to_features(rgb_image,boxes):

    # Input: rgb_images and the resulting bounding boxes from the model
    # Output: Lists of extracted features namely 
    crop_img_list  = []  # images for further testing
    player_x_coord = []  # x-coord of player of each box
    assignment     = []  # main color pigment of each box     
    
    #Compute the hue of grass:
    grass_average = rgb_image[:,:,::-1].mean(axis=0).mean(axis=0)

    # For each box, extract features
    for box in boxes:

        # If box class is a person then proceed
        if box.cls[0].item() == 0:   
            crop = box.xyxy
            (x_1, y_1, x_2, y_2) = np.concatenate(crop.cpu().detach().int().tolist()) # Orgin coordinates
            player_saved = rgb_image[y_1:y_2,x_1:x_2]
            crop_img_list.append(player_saved[:,:,::-1])       # save feature 1
            player_x_coord.append(0.5*(x_1 + x_2))             # save feature 2

            # For the main color, we first reduce area of search to only shirts
            # Further filter to remove all the grass hue from the image
            # With all the accepted pixels, we compute the average color of the shirts
            (new_x_1,new_x_2,new_y_1,new_y_2) = area_reduction(x_1,x_2,y_1,y_2) 
            cropped = rgb_image[new_y_1:new_y_2,new_x_1:new_x_2]
            test_crop = cropped[:,:,::-1].copy()
            accepted_pixel = []
            for row_idx in range(cropped.shape[0]):
                for col_idx in range(cropped.shape[1]):        
                    current_color = test_crop[row_idx][col_idx]
                    if sum(abs(current_color - grass_average)) > 60:
                        accepted_pixel.append(current_color)
            final_color = np.mean(accepted_pixel,axis = 0)
            assignment.append(final_color)                     # save feature 3

    return assignment,player_x_coord,crop_img_list

# Remove the audiences from the image
def pitch_segmentation(rgb_image):
    ## pitch seg
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 20, 50])
    upper_green = np.array([90, 255, 255])

    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    result = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_green)
    temp = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
    h, w, _ = result.shape
    _, green, _ = cv2.split(temp)

    # Find contours
    contours, hierarchy = cv2.findContours(image=green, mode=cv2.RETR_EXTERNAL,
                                      method=cv2.CHAIN_APPROX_NONE)

    contours = max(contours, key=cv2.contourArea)
    rect = np.int16(cv2.boundingRect(contours))
    min_x, min_y, w, h = rect
    max_x, max_y = [min_x + w, min_y + h]
    cv2.rectangle(temp, (min_x, min_y), (max_x, max_y),color=(255, 0, 0), thickness=5)

    # Crop the image
    result = cv2.resize(result[min_y:max_y, min_x:max_x], (w, h))
    rgb_image = cv2.resize(rgb_image[min_y:max_y, min_x:max_x], (w, h))

    return rgb_image

# Convert rgb to CLE_lab for computing the difference between 2 different colors
def rgb_to_lab(rgb_color):
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
    lab_team_1_color = rgb_to_lab(team_1_color)
    lab_team_2_color = rgb_to_lab(team_2_color)
    lab_assignment = [rgb_to_lab(x) for x in assignment]

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
                    updated_labels[x] = team_2_label
                else:
                    updated_labels[x] = misc_label
            
    if len(convert_misc_2_idx) > 0:
            for y in convert_misc_2_idx:
                if delta_e_cie2000(lab_team_1_color,lab_assignment[y]) < 30:
                    updated_labels[y] = team_1_label
                else:
                    updated_labels[y] = misc_label
    
    return updated_labels, lab_team_1_color, lab_team_2_color

# Identify the position of the misc label
def position_text(player_x_coord,new_team_1,new_team_2,new_misc):
    # Input: Players x coordinates and the index of the all classes
    # Output: determine the position of the misc label (Left or Right)

    misc_pos = [player_x_coord[i] for i in new_misc]
    avg_x_team_1 = np.mean([player_x_coord[i] for i in new_team_1])
    avg_x_team_2 = np.mean([player_x_coord[i] for i in new_team_2])

    # Identify left vs right position. The metric can be changed if needed
    if avg_x_team_1 < avg_x_team_2:
        left_coord = avg_x_team_1
    else:
        left_coord = avg_x_team_2
    
    # Update text
    for i in range(len(new_misc)):
        if misc_pos[i] <= left_coord:
            misc_pos[i] = "Left"
        else:
            misc_pos[i] = "Right"
    
    return misc_pos







    