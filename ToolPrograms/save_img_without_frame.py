import sys
sys.path.append('ConeDetection')

import os
import cv2
from ConeDetector import ConeDetector



def remove_frame_from_images(input_folder, output_folder):
    detector = ConeDetector()
    raw_folder = input_folder
    processed_folder = output_folder

    for filename in os.listdir(raw_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(raw_folder, filename)
            img = cv2.imread(filepath)

            img_sliced = detector.remove_image_frame(img)
            processed_filepath = os.path.join(processed_folder, filename)

            cv2.imwrite(processed_filepath, img_sliced)


input_path = "Images\Raw"
output_path = "Images\FrameRemoved"


remove_frame_from_images(input_path, output_path)







