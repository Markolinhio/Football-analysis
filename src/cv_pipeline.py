import os
import sys
from pathlib import Path

import cv2
import numpy as np
import itertools

from ultralytics import YOLO
from cv_utils import *

import segmentation_models_pytorch as smp


def box_center(bounding_box):
    startX, startY, endX, endY = bounding_box
    centerX = (startX + endX) // 2
    centerY = (startY + endY) // 2
    return (centerX, centerY)


def box_bottom_coord(bounding_box):
    startX, startY, endX, endY = bounding_box
    bottom_centerX = (startX + endX) // 2
    bottom_centerY = endY
    return (bottom_centerX, bottom_centerY)


def model_predict(model, frame):

    results = model.predict(frame)
    player_box_list = []
    ball_box = None
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls[0].item() == 0:
                coord = box.xyxy.cpu().detach().int().tolist()[0]
                player_box_list.append(coord)
            elif box.cls[0].item() == 1:
                coord = box.xyxy.cpu().detach().int().tolist()[0]
                ball_box = coord
            else:
                continue
    return player_box_list, ball_box


def player_color_from_frame(player_box_list, frame):
    #Compute the hue of grass:
    grass_average = frame[:,:,::-1].mean(axis=0).mean(axis=0)
    color_list = []

    for box in player_box_list:
        player_color = average_color_from_box(box, frame, grass_average)
        color_list.append(player_color)
    
    return color_list


def cluster_objects_by_color(color_list):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(color_list)
    labels=kmeans.labels_

    # Extract the labels into 2 teams and misc color
    teams = Counter(labels).most_common(3)
    team_1_idx = np.where(labels == teams[0][0])[0]
    team_2_idx = np.where(labels == teams[1][0])[0]
    misc_idx = np.where(labels == teams[2][0])[0]
    return team_1_idx, team_2_idx, misc_idx


def match_color(lab_color_1,lab_color_2):
    threshold = 30
    diff = delta_e_cie2000(lab_color_1,lab_color_2)
    if diff > threshold:
        return False
    else:
        return True 

def fix_annotation_by_color(color_list, team_1_global_color, team_2_global_color,
                             threshold=60):
    team_1_idx, team_2_idx, misc_idx = cluster_objects_by_color(color_list)

    lab_team_1_color = rgb_to_lab_color(team_1_global_color)
    lab_team_2_color = rgb_to_lab_color(team_2_global_color)
    lab_color_list = [rgb_to_lab_color(x) for x in color_list]

    fixed_team_1_idx = []
    fixed_team_2_idx = []
    fixed_misc_idx = []
    for i in range(len(color_list)):
        lab_color_box = lab_color_list[i]
        delta_1 = delta_e_cie2000(lab_team_1_color, lab_color_box)
        delta_2 = delta_e_cie2000(lab_team_2_color, lab_color_box)

        if i in team_1_idx:
            if delta_2 > threshold and delta_2 < 30:
                fixed_team_2_idx.append(i)
            else:
                fixed_misc_idx.append(i)

    return fixed_team_1_idx, fixed_team_2_idx, fixed_misc_idx





def assign_player_by_startingXI(start_box_list, formation_dict):
    return True


def is_same_object(previous_bounding_box, current_bounding_box, threshold=15):
    center_previous_box = box_center(previous_bounding_box)
    center_current_box = box_center(current_bounding_box)

    distance = np.sqrt((center_previous_box[0] - center_current_box[0])**2 + 
                       (center_previous_box[1] - center_current_box[1])**2)
    
    if distance < threshold:
        return True
    else:
        return False
    

def visualize_team_players(frame, team_1_box, team_2_box=None, misc_box=None, ball_box=None):
    temp = frame.copy()
    for player_box in team_1_box:
        center = box_center(player_box)
        cv2.circle(temp, center, 7, (0, 255, 0), -1)
        bottom = box_bottom_coord(player_box)
        cv2.circle(temp, bottom, 7, (0, 255, 0), -1)
        cv2.rectangle(temp, (player_box[0], player_box[1]), (player_box[2], player_box[3]),
                    (255, 0, 0), 3)

    if team_2_box:    
        for player_box in team_2_box:
            center = box_center(player_box)
            cv2.circle(temp, center, 7, (0, 255, 0), -1)
            bottom = box_bottom_coord(player_box)
            cv2.circle(temp, bottom, 7, (0, 255, 0), -1)
            cv2.rectangle(temp, (player_box[0], player_box[1]), (player_box[2], player_box[3]),
                        (0, 0, 255), 3)
    
    if misc_box:
        for player_box in misc_box:
            center = box_center(player_box)
            cv2.circle(temp, center, 7, (0, 255, 0), -1)
            bottom = box_bottom_coord(player_box)
            cv2.circle(temp, bottom, 7, (0, 255, 0), -1)
            cv2.rectangle(temp, (player_box[0], player_box[1]), (player_box[2], player_box[3]),
                        (255, 0, 255), 3)

    if ball_box:
        cv2.rectangle(temp, (ball_box[0], ball_box[1]), (ball_box[2], ball_box[3]),
                    (255, 0, 0), 3)
        bottom = box_bottom_coord(ball_box)
        cv2.circle(temp, bottom, 7, (0, 255, 0), -1)

    plt.figure()
    plt.imshow(temp)