import cv2
import numpy as np

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


            conditions = [features[1] > thresholds["min_area"], 
                            features[2] > thresholds["min_perimeter"], 
                            features[4] > thresholds["min_aspect_ratio"], 
                            features[5] > thresholds["min_extent"],
                            features[5] < thresholds["max_extent"],                
                            features[1] > thresholds["min_area"],
                            features[13] > thresholds["solidity"],
                            CoM_ratio < thresholds["mass_center_height_ratio"]]


            if 0 not in conditions:
                approx = cv2.approxPolyDP(blob, 3, True) #0.015*features[2]
                bbox = cv2.boundingRect(approx)

                #Increas height of detected cones, as the detecion only sees the bottom
                cone_height_scale = 1.2

                #Increase height of bounding box to not only contain the bottom part, but also the top part
                bbox = list(bbox)
                bbox[1] = int(bbox[1]-bbox[3]*1.2)
                bbox[3] = int(bbox[3]+bbox[3]*1.2)

                
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

        

        
        
        

