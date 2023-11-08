import sys
sys.path.append('ConeDetection')

import os
import cv2
from image_processor import ImageProcessor



def remove_frame_from_images(input_folder, output_folder, num_of_images=-1, border_thickness = None):
    Processor = ImageProcessor()
    raw_folder = input_folder
    processed_folder = output_folder

    for filename in os.listdir(raw_folder)[:num_of_images]:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(raw_folder, filename)
            img = cv2.imread(filepath)

            img_sliced = Processor.remove_image_frame(img, border_thickness=border_thickness)
            processed_filepath = os.path.join(processed_folder, filename)

            cv2.imwrite(processed_filepath, img_sliced)


input_path = "YOLO\\fsoco_sample\\fsoco_sample\\images"
output_path = "YOLO\\data\\images\\train"


#remove_frame_from_images(input_path, output_path, border_thickness = 140)







