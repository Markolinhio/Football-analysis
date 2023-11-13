import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
plt.rcParams["figure.figsize"] = (24,18)
from cv_utils import *

model_path = os.path.join(os.getcwd(), 'models')
print(model_path)
model = YOLO(os.path.join(model_path, "yolov8n_2nd_train.pt"))
data_path = os.path.join(os.getcwd(), 'data')
vid_name =  'real_test'

# Get several images to test
images_path = os.path.join(data_path, 'images' + '/' + vid_name)  #print(os.path.exists(images_path))
destination_path_coco = os.path.join(data_path, 'coco_datasets' +'/'+  vid_name)

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

for image_name in os.listdir(images_path)[:2]:
    
    # Crop out the audiences via pitch segmentation
    image_ad  = os.path.join(images_path, image_name)
    rgb_image = cv2.imread(image_ad, cv2.COLOR_BGR2RGB)
    rgb_image = pitch_segmentation(rgb_image)
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
    
            
    

#print([x["category_id"] for x in all_bbox_info])
# Write COCO dict
coco_dict["info"] = coco_info
coco_dict["license"] = {'name': vid_name,
                            'id': '',
                            'url': ''}
coco_dict["categories"] = categories_dict
coco_dict["images"] = all_image_info
coco_dict["annotations"] = all_bbox_info

with open(os.path.join(destination_path_coco, 'instances_default.json'), 'w') as coco:
   json.dump(coco_dict, coco)

print("Done")
