import sys
#sys.path.append('ConeDetection')

import os
import cv2
import numpy as np

from ConeDetector import ConeDetector

Detector = ConeDetector()


# if you want it to process the image for blue cones, set Blue to True, else set it to False
def process_image(input_path, Blue, name):
    # Read image
    img = cv2.imread(f"{input_path}", cv2.IMREAD_COLOR)

    # Apply function to image
    if Blue == True:
        processed_img = Detector.find_blue_cones(img)
        #_, processed_img = Detector.colour_threshold_HSV(img, name + "_blue" , [80,95,110], [165,255,255])
    else:
        _, processed_img = Detector.colour_threshold_HSV(img, name + "_yellow", [20,95,110], [80,255,255])

    return processed_img

#creates a folder for the processed images, that originates from the input folder,
#and saves them in the new folder.
def process_folder(input_folder, output_folder):
    # Loop through all files in input folder
    for filename in os.listdir(input_folder):
        # Check if file is an image
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Get input and output paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder)

            # Create directory if it does not exist
            if not os.path.exists(input_path):
                os.makedirs(input_path)
                
            if not os.path.exists(output_path):
                os.makedirs(output_path) 

            # Process image and save to output folder
            processed_imgBlue = process_image(input_path, True, filename)
            processed_imgYellow = process_image(input_path, False, filename)
            
            cv2.imwrite(output_path + f'\\blue_{filename}', processed_imgBlue)
            cv2.imwrite(output_path + f'\\yellow_{filename}', processed_imgYellow)


input_path = "Images\FrameRemoved"
output_path = "Images\BinaryImages_img"

process_folder(input_path, output_path)