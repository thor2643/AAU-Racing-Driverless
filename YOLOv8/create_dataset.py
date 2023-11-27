import os
import cv2
import sys
sys.path.append('ToolPrograms')
sys.path.append('YOLO')

import save_img_without_frame
import label_converter

data_input_path = "fsoco_bounding_boxes_train"
data_output_path = "YOLO\\data"

class_names = ["yellow_cone", "blue_cone", "orange_cone", "large_orange_cone", "unknown_cone"]

imgs_per_folder = 10 #set to -1 for all

for folder in os.listdir(data_input_path):
        save_img_without_frame.remove_frame_from_images(f"{data_input_path}\\{folder}\\img", 
                                                        f"{data_output_path}\\images\\train", 
                                                        num_of_images = imgs_per_folder,
                                                        border_thickness = 140)
    
        label_converter.convert_supervisely_to_yolo(f"{data_input_path}\\{folder}\\ann",
                                                    f"{data_output_path}\\labels\\train",
                                                    class_names = class_names,
                                                    num_of_files = imgs_per_folder)

"""
for filename in os.listdir(raw_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join(raw_folder, filename)
        img = cv2.imread(filepath)

        img_sliced = Processor.remove_image_frame(img, border_thickness=border_thickness)
        processed_filepath = os.path.join(processed_folder, filename)

        cv2.imwrite(processed_filepath, img_sliced)
"""