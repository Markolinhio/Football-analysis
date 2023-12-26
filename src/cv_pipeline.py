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


def calculate_distance(box1, box2):
    """
    Calculate the Euclidean distance between the centers of two bounding boxes.
    """
    return np.linalg.norm(box_center(box1) - box_center(box2))


def box_intersection(bounding_box_1, bounding_box_2):
    dx = min(bounding_box_1[2], bounding_box_2[2]) - max(bounding_box_1[0], bounding_box_2[0])
    dy = min(bounding_box_1[3], bounding_box_2[3]) - max(bounding_box_1[1], bounding_box_2[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return -1


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


def filter_boxes_by_obscurity(box_list, threshold=2000):
    # Get the average area of player boxes
    box_area_list = [(box[3] - box[1])*(box[2] - box[0]) for box in box_list]
    box_area_average = np.median(box_area_list)
    filtered_box_list = []
    for box in box_list:
        box_area = (box[3] - box[1])*(box[2] - box[0])
        if box_area > threshold:
            filtered_box_list.append(box)
    return filtered_box_list


def player_color_from_frame(player_box_list, frame, grass_mask):
    # Remove black from mask
    con1 = grass_mask[:,:,0] != 0
    con2 = grass_mask[:,:,1] != 0
    con3 = grass_mask[:,:,2] != 0
    grass_mask = grass_mask[np.where(con1 & con2 & con3)]
    # Compute the hue of grass:
    grass_average = grass_mask.mean(axis=0)
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


def match_color(lab_color_1, lab_color_2, threshold=30):
    diff = delta_e_cie2000(lab_color_1,lab_color_2)
    if diff > threshold:
        return False
    else:
        return True 

def get_teams_average_color(color_list):
    team_1_idx, team_2_idx, misc_idx = cluster_objects_by_color(color_list)
    team_1_color = np.mean([color_list[i] for i in team_1_idx], axis=0)
    team_2_color = np.mean([color_list[i] for i in team_2_idx], axis=0)

    return team_1_color, team_2_color


def fix_annotation_by_color(color_list, team_1_global_color, team_2_global_color,
                            threshold=30):
    team_1_idx, team_2_idx, misc_idx = cluster_objects_by_color(color_list)

    lab_team_1_color = rgb2lab(team_1_global_color)
    lab_team_2_color = rgb2lab(team_2_global_color)
    lab_color_list = [rgb2lab(x) for x in color_list]

    fixed_team_1_idx = []
    fixed_team_2_idx = []
    fixed_misc_idx = []
    for i in range(len(color_list)):
        if match_color(lab_color_list[i], lab_team_1_color, threshold):
            fixed_team_1_idx.append(i)
        elif match_color(lab_color_list[i], lab_team_2_color, threshold):
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
    

def in_possession(ball_box, player_box, intersection_threshold=10, distance_threshold=10):
    intersection_area = box_intersection(ball_box, player_box)
    distance = np.linalg.norm(np.array(box_center(ball_box)) - np.array(box_center(player_box)))

    if intersection_area <= intersection_threshold and distance <= distance_threshold:
        return True
    else:
        return False
    

def distance_from_ball(ball_box, player_box):
    return np.linalg.norm(np.array(box_center(ball_box)) - np.array(box_center(player_box)))

def check_possession(team_1_box, team_2_box, ball_box, current_possession):
    """
    Check which team possesses the ball
    If no player possesses the ball or there are no balls in the frame, return the possession of the previous frame (current_possession)
    If many players are closed to the ball, possession belongs to the closest one
    
    Return:
        Possession: 1 or 2
    """
    if ball_box==None:
        return current_possession
    
    possession = None
    min_dist = 99999999999999
    for player_box in team_1_box:
        if in_possession(ball_box, player_box):
            if distance_from_ball(ball_box, player_box) < min_dist:
                possession = 1
                min_dist = distance_from_ball(ball_box, player_box)
            
    for player_box in team_2_box:
        if in_possession(ball_box, player_box):
            if distance_from_ball(ball_box, player_box) < min_dist:
                possession = 2
                min_dist = distance_from_ball(ball_box, player_box)
            
    if possession==None:
        return current_possession
    
    return possession
    


def in_collision(player_box_1, player_box_2, intersection_threshold=10, distance_threshold=10):
    intersection_area = box_intersection(player_box_1, player_box_2)
    distance = np.linalg.norm(np.array(box_bottom_coord(player_box_1)) - np.array(box_bottom_coord(player_box_2)))

    if intersection_area <= intersection_threshold and distance <= distance_threshold:
        return True
    else:
        return False
    

def check_collision(team_1_box, team_2_box, misc_box):
    """
    If 2 player boxes collides then remove both
    If a player box collides with a misc box then remove the misc one
    
    Return:
        team_1_uncollided
        team_2_uncollided
        misc_uncollided
    """
    # Check within a team
    team_1_uncollided = [
        box 
            for i, box in enumerate(team_1_box) 
                if all(not in_collision(box, other_box) 
                    for other_box in team_1_box[:i] + team_1_box[i+1:])
    ]
    team_2_uncollided = [
        box 
            for i, box in enumerate(team_2_box) 
                if all(not in_collision(box, other_box) 
                    for other_box in team_2_box[:i] + team_2_box[i+1:])
    ]
    
    # Check between 2 teams
    team_1_uncollided = [a for a, b in zip(team_1_uncollided, team_2_uncollided) if not in_collision(a, b)]
    team_2_uncollided = [b for a, b in zip(team_1_uncollided, team_2_uncollided) if not in_collision(a, b)]
    
    # Check misc if collided with players
    misc_uncollided = [a for a, b in zip(misc_box, team_1_uncollided+team_2_uncollided) if not in_collision(a, b)]
    
    return team_1_uncollided, team_2_uncollided, misc_uncollided


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
                    (0, 255, 0), 3)
        bottom = box_bottom_coord(ball_box)
        cv2.circle(temp, bottom, 7, (0, 255, 0), -1)

    plt.figure()
    plt.imshow(temp)


