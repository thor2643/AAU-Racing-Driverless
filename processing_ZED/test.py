#from PIL import Image
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import time
import matplotlib.pyplot as plt

img = cv2.imread('processing_ZED//hii.png')

def remove_all_but_concrete( img1):
    img = img1.copy()
    lower_HSV = [110, 60, 100]
    upper_HSV= [200, 200, 255]    
    lower_HSV = np.array(lower_HSV, dtype=np.uint8)  # Convert to NumPy array
    upper_HSV = np.array(upper_HSV, dtype=np.uint8)  # 1Convert to NumPy array

    # 1. Convert the image to HSV color space
    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Apply HSV thresholding to find concrete areas
    HSV_mask = cv2.inRange(HSV_img, lower_HSV, upper_HSV)
    
    cv2.imshow('HSV_mask',HSV_mask)
    cv2.waitKey(0)
    
    # 3. Find contours in the binary mask
    contours, _ = cv2.findContours(HSV_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Initialize variables to keep track of the largest blob and its bounding box
    largest_blob = None
    largest_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_blob = contour

    if largest_area > 5000:
        return 0


    if largest_blob is not None:
        # 5. Get the bounding box of the largest blob
        _ , y, _, _ = cv2.boundingRect(largest_blob)

        # 6. Crop the original image using the bounding box
        #concrete_area = img[y-5:y+5+h, x:x+w]
        
        return y
    
y = remove_all_but_concrete(img)
