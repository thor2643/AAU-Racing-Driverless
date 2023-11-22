import cv2
import random
import numpy as np
import os
import shutil

def SV_scaling(img_path, YOLO_label_path, img_output_path, label_output_path):
    img_bgr = cv2.imread(img_path)
    colour_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) 

    scale_S = random.uniform(0.8, 1.2)
    scale_V = random.uniform(0.8, 1.2)
    scale_H = random.uniform(0.9, 1.1)
    
    colour_img[:,:,0] = cv2.convertScaleAbs(colour_img[:,:,0], alpha=scale_H)
    colour_img[:,:,1] = cv2.convertScaleAbs(colour_img[:,:,1], alpha=scale_S) #np.ndarray.astype(colour_img[:,:,1] * 0.2, np.uint8) 
    colour_img[:,:,2] = cv2.convertScaleAbs(colour_img[:,:,2], alpha=scale_V) 

    colour_img = cv2.cvtColor(colour_img, cv2.COLOR_HSV2BGR) 

    shutil.copyfile(YOLO_label_path, label_output_path)

    cv2.imwrite(img_output_path, colour_img)


def horizontal_flip(img_path, YOLO_label_path, img_output_path, label_output_path):
    img_bgr = cv2.imread(img_path)
    img_flipped = cv2.flip(img_bgr, 1)

    with open(YOLO_label_path, 'r') as f:
        lines = f.readlines()

    flipped_lines = []
    for line in lines:
        parts = line.split()
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        flipped_x_center = 1.0 - x_center
        flipped_y_center = y_center
        flipped_width = width
        flipped_height = height

        flipped_line = f"{parts[0]} {flipped_x_center} {flipped_y_center} {flipped_width} {flipped_height}\n"
        flipped_lines.append(flipped_line)

    with open(label_output_path, 'w') as f:
        f.writelines(flipped_lines)

    cv2.imwrite(img_output_path, img_flipped)

    return img_flipped


img_input_folder_path = "Images\OwnData\YOLO_Images\Images"#\\amz_00000.jpg"
img_output_folder_path = "Images\OwnData\YOLO_Images\Images"#\\amz_00000_flipped.jpg"

label_input_folder_path = "Images\OwnData\YOLO_Images\YOLOLabels"#\\amz_00000.txt"
label_output_folder_path = "Images\OwnData\YOLO_Images\YOLOLabels"#\\label.txt"

number_of_files = -1 #-1 for all

#horizontal_flip(img_input_path, label_input_path, img_output_path, label_output_path)


if number_of_files == -1:
    number_of_files = len(os.listdir(img_input_folder_path))
    
for img, label in zip(os.listdir(img_input_folder_path)[:number_of_files], os.listdir(label_input_folder_path)[:number_of_files]):

    img_input_path = img_input_folder_path + "\\" + img
    label_input_path = label_input_folder_path + "\\" + label

    img_output_path = img_output_folder_path + "\\" + img.split(".")[0] + "_flipped.jpg"
    label_output_path = label_output_folder_path + "\\" + label.split(".")[0] + "_flipped.txt"

    horizontal_flip(img_input_path, label_input_path, img_output_path, label_output_path)

    img_output_path = img_output_folder_path + "\\" + img.split(".")[0] + "_SVscaled.jpg"
    label_output_path = label_output_folder_path + "\\" + label.split(".")[0] + "_SVscaled.txt"

    SV_scaling(img_input_path, label_input_path, img_output_path, label_output_path) 


