import cv2
import os
import numpy as np
from image_processor import ImageProcessor

class ConeDetector:
    def __init__(self) -> None:
        pass

    def get_blobs_and_features(self, img, method = cv2.CHAIN_APPROX_SIMPLE):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, method)

        cont_props= []
        i = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            convexity = cv2.isContourConvex(cnt)
            x1,y1,w,h = cv2.boundingRect(cnt)
            x2 = x1+w
            y2 = y1+h
            aspect_ratio = float(w)/h
            rect_area = w*h
            extent = float(area)/rect_area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area)/hull_area
            else:
                solidity = 0

            # Add parameters to list
            #Idx:  0     1             2              3                 4                   5          6  7            8           9  10  11  12      13   
            add = i+1, area, round(perimeter, 1), convexity, round(aspect_ratio, 3), round(extent, 3), w, h, round(hull_area, 1), x1, y1, x2, y2, solidity
            cont_props.append(add)
            i += 1

        return contours, cont_props
    
    def sort_blobs(self, thresholds, blobs, blob_features):
        blobs_bbox_CoM = []

        for blob, features in zip(blobs, blob_features):
            #Used to find center of mass
            M = cv2.moments(blob)

            # calculate x,y coordinate of center
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                CoM_ratio = (features[12]-cY) / features[7]         #(features[12] - (features[10]+cY)) / features[7] 

            except ZeroDivisionError:
                CoM_ratio = 0
                cX = 0
                cY = 0


            conditions = [features[1] > thresholds["min_area"], 
                            features[2] > thresholds["min_perimeter"], 
                            features[4] > thresholds["min_aspect_ratio"], 
                            features[5] > thresholds["min_extent"],
                            features[5] < thresholds["max_extent"],                
                            features[1] > thresholds["min_area"],
                            features[13] > thresholds["solidity"],
                            CoM_ratio < thresholds["mass_center_height_ratio"]]


            #if 0 not in conditions:
            approx = cv2.approxPolyDP(blob, 3, True) #0.015*features[2]
            bbox = cv2.boundingRect(approx)

                #Increas height of detected cones, as the detecion only sees the bottom
                #cone_height_scale = 1.2

                #Increase height of bounding box to not only contain the bottom part, but also the top part
                #bbox = list(bbox)
                #bbox[1] = int(bbox[1]-bbox[3]*1.2)
                #bbox[3] = int(bbox[3]+bbox[3]*1.2)

                
            blobs_bbox_CoM.append([features, bbox, [cX, cY]])
        
        return blobs_bbox_CoM
    

    def remove_cone_top_blobs(self, blobs_bbox_CoM):
        #Check if CoM's are inside other bounding boxes to determine and remove the bbox of the top part of the cone
        i = 0
        while i < len(blobs_bbox_CoM):
            #print()
            #print("Herfra")
            #print("{}: {}".format(blobs_bbox_CoM[i][0][0], blobs_bbox_CoM[i][2]))
            j =0
            while j < len(blobs_bbox_CoM):
                x_min = blobs_bbox_CoM[j][1][0]
                x_max = blobs_bbox_CoM[j][1][0] + blobs_bbox_CoM[j][1][2]
                y_min = blobs_bbox_CoM[j][1][1]
                y_max = blobs_bbox_CoM[j][1][1] + blobs_bbox_CoM[j][1][3]

                #print("{}: {}".format(blobs_bbox_CoM[j][0][0], blobs_bbox_CoM[j][1]))
                if x_min < blobs_bbox_CoM[i][2][0] < x_max and y_min < blobs_bbox_CoM[i][2][1] < y_max and not blobs_bbox_CoM[i][0][0]==blobs_bbox_CoM[j][0][0]:
                    del blobs_bbox_CoM[i]
                    break
                j+=1

            i+=1
        
        return blobs_bbox_CoM

    def draw_bbox_CoM(self, img, blobs_bbox_CoM):
        for blob in blobs_bbox_CoM:
            cv2.putText(img, '{}'.format(blob[0][0]), (blob[0][9] + 30, blob[0][10] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,(255, 0, 255), 2)
            cv2.rectangle(img, (int(blob[1][0]), blob[1][1]), 
                                    (int(blob[1][0]+blob[1][2]), int(blob[1][1]+blob[1][3])), (255, 0, 255), 2)
            cv2.circle(img, (blob[2][0], blob[2][1]), 2, (0, 0, 255), -1)

        return img
    
    def colour_threshold(self, img_BGR, lower_val: list, upper_val: list, colourspace="HSV"):
        if colourspace == "HSV":
            colour_img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV) 
        elif colourspace == "BGR":
            colour_img = img_BGR
        elif colourspace == "YUV":
            colour_img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV)

        # set lower and upper colour limits
        temp_lower_val = np.array(lower_val)
        temp_upper_val = np.array(upper_val)

        # Threshold the image to get a desired colour range
        mask = cv2.inRange(colour_img, temp_lower_val, temp_upper_val)

        # apply mask to original image. This shows the desired colour range with a black blackground
        desired_colours = cv2.bitwise_and(img_BGR, img_BGR, mask = mask)

        # create a white image with the dimensions of the input image. Scale with 255 to set all pixel values to 255 instead of 1, because that would result in a black image.
        background = np.ones(img_BGR.shape, img_BGR.dtype)*255

        # invert the mask that blocks everything except the desired colour range
        mask_inv = cv2.bitwise_not(mask)
        # apply the inverted mask to the white image
        masked_background = cv2.bitwise_and(background, background, mask = mask_inv)
        # add the 2 images together. This yields an image with white background and the desired colours shown
        final = cv2.add(desired_colours, masked_background)
        
        #show image
        #cv2.imshow(name, final)
        return final, mask

    def find_blue_cones(self, img):
        # Find all blue parts of the cone using HSV colour thresholding
        img_blue, _ = self.colour_threshold(img, [80,95,110], [165,255,255])
        
        # Find all the white lines on the cones by inverting the image to easily detect white colours with HSV colour thresholding
        inv_img = 255-img
        img_white_lines, _ = self.colour_threshold(inv_img, [0,0,0], [255,255,55])

        # Convert the images to greyscale to convert them to binary images
        gray_img_blue = cv2.cvtColor(img_blue, cv2.COLOR_BGR2GRAY)
        gray_img_white_lines = cv2.cvtColor(img_white_lines, cv2.COLOR_BGR2GRAY)

        # Convert the greyscaled images to binary images. 
        _, binary_img_blue = cv2.threshold(gray_img_blue, 210, 255, cv2.THRESH_BINARY)
        _, binary_img_white_lines = cv2.threshold(gray_img_white_lines, 140, 255, cv2.THRESH_BINARY)

        # The bitwise_and operator helps us combine the to images so that the white colours (now black blobs) that were previously missing from the blue cones (also black blobs) 
        # are combined with each other resulting in an image consisting of whole cones
        result = cv2.bitwise_and(binary_img_blue, binary_img_white_lines)
        result_img = 255-result

        # Create a kernel to apply opening (dilation) and closing (erosion) to the image, which will help connecting the black and white cone parts completely, 
        # as there are still a few pixels that need to be connected
        kernel = np.ones((3,3), np.uint8)
        opening_img = cv2.dilate(result_img, kernel, iterations= 1)
        closing_img = cv2.erode(opening_img, kernel, iterations= 1)
        
        cv2.imshow("Blue Cones", closing_img)
        # Return the image
        return closing_img

    def find_yellow_cones_with_laplacian(self, img):
        # Threshold images
        img_yellow = self.colour_threshold(img, [20, 95, 110], [35, 255, 255])
        img_cone = self.colour_threshold(img, [26, 32, 42], [100, 255, 255], "BGR")
    
        # Convert to grayscale and then to binary
        gray_img_cone = cv2.cvtColor(img_cone, cv2.COLOR_BGR2GRAY)
        _, binary_img_cone = cv2.threshold(gray_img_cone, 245, 255, cv2.THRESH_BINARY)
    
        BGR_img_yellow = cv2.cvtColor(img_yellow, cv2.COLOR_HSV2BGR) 
        
        # Apply Laplacian to img_yellow
        laplacian = cv2.Laplacian(BGR_img_yellow, cv2.CV_64F)
        laplacian_2 = cv2.convertScaleAbs(laplacian) 
        #print(laplacian_2.shape)
        # Resize Laplacian to match the dimensions of binary_img_cone
        gray_img_laplacian = cv2.cvtColor(laplacian_2, cv2.COLOR_BGR2GRAY)

        # Apply threshold to Laplacian result
        _, binary_img_laplacian = cv2.threshold(gray_img_laplacian, 100, 255, cv2.THRESH_BINARY)
        inv_binary_img_laplacian = 255-binary_img_laplacian
        
        # Perform bitwise_and operation with the binary cone mask
        summed_img = cv2.bitwise_and(binary_img_cone, inv_binary_img_laplacian)

        # Use opening and closing on the image to complete the cones
        kernel = np.ones((3,3), np.uint8)
        opening_img = cv2.erode(summed_img, kernel, iterations= 1)
        closing_img = cv2.dilate(opening_img, kernel, iterations= 1)

        # Invert the image to get black cones on a white background and remove any larger blobs that are not cones
        inv_closing_img = 255 - closing_img
        final_img = ImageProcessor.remove_blobs(inv_closing_img)
        
        # Display the result
        cv2.imshow('Yellow Cones', final_img)
        return final_img
    
    def find_orange_cones(self, img):
        img_orange_cones = self.colour_threshold(img, [0, 0, 81], [163, 188, 255], "BGR")

        gray_img_orange_cones = cv2.cvtColor(img_orange_cones, cv2.COLOR_BGR2GRAY)

        _, binary_img_orange_cones = cv2.threshold(gray_img_orange_cones, 165, 255, cv2.THRESH_BINARY)  

        cv2.imshow('Binary Orange Cones', binary_img_orange_cones)