import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import time

def finds_LAB_reference_from_folder(folder):
    numb_images = 0
    L_s_mean = 0
    L_s_std = 0
    A_s_mean = 0
    A_s_std = 0
    B_s_mean = 0
    B_s_std = 0
    for filename in os.listdir(folder):
        if filename.endswith(".jpg" or ".png"):
            image = cv2.imread(os.path.join(folder, filename))
            #convert the images from RGB to LAB
            img_source = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
            #splitting the images into channels
            L_s ,A_s ,B_s =cv2.split(img_source)
            #computing the mean and standard deviation of each channel for both images
            #for source image
            L_s_mean_temp, L_s_std_temp = L_s.mean(), L_s.std()
            A_s_mean_temp, A_s_std_temp = A_s.mean(), A_s.std()
            B_s_mean_temp, B_s_std_temp = B_s.mean(), B_s.std()
            
            #add it together for all images
            L_s_mean += L_s_mean_temp
            L_s_std += L_s_std_temp
            A_s_mean += A_s_mean_temp
            A_s_std += A_s_std_temp
            B_s_mean += B_s_mean_temp
            B_s_std += B_s_std_temp
            
            numb_images += 1
    #calculate the mean value of all images
    L_s_mean = L_s_mean / numb_images
    L_s_std = L_s_std / numb_images
    A_s_mean = A_s_mean / numb_images
    A_s_std = A_s_std / numb_images
    B_s_mean = B_s_mean / numb_images
    B_s_std = B_s_std / numb_images
    return L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std

def color_transfer(processed_img, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std):
    #convert the images from RGB to LAB
    img_target = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB).astype("float32")
    #splitting the images into channels
    L_t ,A_t ,B_t =cv2.split(img_target)
    
    #for target image
    L_t_mean, L_t_std = L_t.mean(), L_t.std()
    A_t_mean, A_t_std = A_t.mean(), A_t.std()
    B_t_mean, B_t_std = B_t.mean(), B_t.std()

    #subtracting the means from the target image
    L_t -= L_t_mean
    A_t -= A_t_mean
    B_t -= B_t_mean
    
    #scaling the target channels by the standard deviation ratio
    L_t = L_t * (L_t_std / L_s_std)
    A_t = A_t * (A_t_std / A_s_std)
    B_t = B_t * (B_t_std / B_s_std)
    
    #Add in the means of the Lab channels for the source.
    L_t += L_s_mean
    A_t += A_s_mean
    B_t += B_s_mean
    
    #Making shure all values are in the range [0, 255]
    L_t = np.clip(L_t, 0, 255)
    A_t = np.clip(A_t, 0, 255)
    B_t = np.clip(B_t, 0, 255)
    
    # Merge the channels together and convert back to the RGB color space,
    # being sure to utilize the 8-bit unsigned integer data type.
    transfer = cv2.merge([L_t, A_t, B_t])
    temp_img = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    return temp_img

def color_enhancement(processed_img):
    # Convert BGR to RGB colorspace
    image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    # Convert to PIL image
    img = Image.fromarray(image_rgb)
    # Creating an object of Color class
    img = ImageEnhance.Color(img)
    # this factor controls the enhancement factor. 0 gives a black and white image. 1 gives the original image
    enhanced_image = img.enhance(2)
    # Convert the enhanced image to an OpenCV format
    temp_img = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
    return temp_img

def find_yellow(processed_img):

    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)

    # Define range of yellow color in HSV
    lower_yellow = np.array([20, 62, 135]) 
    upper_yellow = np.array([31, 255, 247])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    temp_img = cv2.bitwise_and(processed_img, processed_img, mask=mask) 
    return temp_img
    
def find_blue(processed_img):

    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV 
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    temp_img = cv2.bitwise_and(processed_img, processed_img, mask=mask) 
    return temp_img

def create_templates2():
    #reading all the templates
    blue_template_start = cv2.imread("preproccesing//preprocesing_img//blue_template.jpg")
    yellow_template_start = cv2.imread("preproccesing//preprocesing_img//yellow_template1.jpg")
    #making the yellow template the same size as the blue template
    yellow_template_start = cv2.resize(yellow_template_start, (int(37), int(55)))
    
    #creating the templates with different sizes
    blue_template = cv2.resize(blue_template_start, (int(37* 3.75), int(55*3.75)))
    yellow_template = cv2.resize(yellow_template_start, (int(37* 3.75), int(55*3.75)))
    
    blue_template1 = cv2.resize(blue_template_start, (int(37* 2.75), int(55*2.75)))
    yellow_template1 = cv2.resize(yellow_template_start, (int(37* 2.75), int(55*2.75)))
    
    blue_template2 = cv2.resize(blue_template_start, (int(37* 1.75), int((55* 1.75)*2)))
    yellow_template2 = cv2.resize(yellow_template_start, (int(37*1.75), int((55*1.75)*2)))
   
    blue_template3 = cv2.resize(blue_template_start, (int(37* 2.25), int(55*2.25)))
    yellow_template3 = cv2.resize(yellow_template_start, (int(37* 2.75), int(55*2.75)))
    
    blue_template4 = cv2.resize(blue_template_start, (int(37* 1.3), int((55* 1.3)*2)))
    yellow_template4 = cv2.resize(yellow_template_start, (int(37*1.3), int((55*1.3)*2))) 
    
    blue_template5 = cv2.resize(blue_template_start, (int(37* 1.75), int(55*1.75)))
    yellow_template5 = cv2.resize(yellow_template_start, (int(37* 1.75), int(55*1.75)))
    
    blue_template6 = blue_template_start.copy()
    yellow_template6 = yellow_template_start.copy()
    
    blue_template7 = cv2.resize(blue_template_start, (int(37* 1), int((55* 1)*2)))
    yellow_template7 = cv2.resize(yellow_template_start, (int(37*1), int((55*1)*2))) 
    
    blue_template8 = cv2.imread("preproccesing//preprocesing_img//blue_template1.jpg")
    yellow_template8 = cv2.imread("preproccesing//preprocesing_img//yellow_template.jpg")
    
    blue_template9 = cv2.resize(blue_template_start, (int(37* 1.25), int(55*1.25)))
    yellow_template9 = cv2.resize(yellow_template_start, (int(37* 1.25), int(55*1.25)))
    
    blue_template10 = cv2.resize(blue_template_start, (int(37* 0.75), int((55* 0.75)*2)))
    yellow_template10 = cv2.resize(yellow_template_start, (int(37*0.75), int((55*0.75)*2)))
    
    blue_template11 = cv2.resize(blue_template_start, (int(37* 0.75), int(55*0.75)))
    yellow_template11 = cv2.resize(yellow_template_start, (int(37* 0.75), int(55*0.75)))

    blue_template12 = cv2.resize(blue_template_start, (int(37* 0.5), int(55*0.5)))
    yellow_template12 = cv2.resize(yellow_template_start, (int(37* 0.5), int(55*0.5)))
    
    blue_template13 = cv2.imread("preproccesing//preprocesing_img//blue_template2.jpg")
    yellow_template13 = cv2.imread("preproccesing//preprocesing_img//yellow_template2.jpg")
    
    blue_template14 = cv2.imread("preproccesing//preprocesing_img//blue_template3.jpg")
    yellow_template14 = cv2.imread("preproccesing//preprocesing_img//yellow_template3.jpg")

    templates = [yellow_template, yellow_template1, yellow_template2, yellow_template3, yellow_template4, yellow_template5, yellow_template6, yellow_template7, yellow_template8, yellow_template9, yellow_template10, yellow_template11, yellow_template12, yellow_template13, yellow_template14, blue_template, blue_template1, blue_template2, blue_template3, blue_template4, blue_template5, blue_template6, blue_template7, blue_template8, blue_template9, blue_template10, blue_template11, blue_template12, blue_template13, blue_template14]
    return templates

def template_matching(frame, templates):
    #Variables
    c = 0
    cone_number = [(0),(0)]
    allowed_distance=30   #pixels
    new_cone = True
    distance=0
    filtered_cones= [[], []]
    width_height = [[],[]]
    i = 0  
    thresholds_for_detections = []
    
    frame_copy = frame.copy()
   
    for i, template in enumerate(templates):
        if i >= 10:
            threshold = 0.75
        else:
            threshold = 0.7
        if i == len(templates)/2:
            c = 1
            new_cone = True
                
        w, h = template.shape[1], template.shape[0]
        res = cv2.matchTemplate(frame_copy,template,cv2.TM_CCOEFF_NORMED)  
        loc = np.where( res >= threshold)
        # Iterate through the detections and store the threshold value for each
        for pt in zip(*loc[::-1]):
            threshold_value = res[pt[1], pt[0]]  # Get the threshold value at the detection point
            thresholds_for_detections.append(threshold_value)
        
        #Sorting out the cones that are too close to each other based on the threshold value
        for a, pt in enumerate(zip(*loc[::-1])):
            if cone_number[c] != 0:
                #the following part sorts the found cones out, so that only one cone is found in each location
                for u in range(cone_number[c]):
                    distance = np.sqrt((pt[0] - filtered_cones[c][u][0]) ** 2 + (pt[1] - filtered_cones[c][u][1]) ** 2)
                    if distance > allowed_distance:
                        new_cone = True
                    elif distance < allowed_distance and thresholds_for_detections[a] < filtered_cones[c][u][2]:
                        new_cone = False
                        break #it stops looking
                    elif distance < allowed_distance and thresholds_for_detections[a] >= filtered_cones[c][u][2]:
                        #if the new cone has a better threshold than the old one, the old one is replaced
                        del filtered_cones[c][u]
                        del width_height[c][u]
                        cone_number[c] -= 1
                        new_cone = True
                        break #it stops looking
                        
            if new_cone == True:
                cone_number[c] += 1
                ptk = [pt[0], pt[1], thresholds_for_detections[a]]
                filtered_cones[c].append(ptk)
                width_height[c].append([w, h])
                    
    return frame, filtered_cones, width_height 

"""
        #Sorting out the cones that are too close to each other
        for pt in zip(*loc[::-1]):
            if cone_number[c] != 0:
                #the following part sorts the found cones out, so that only one cone is found in each location
                for u in range(cone_number[c]):
                    distance = np.sqrt((pt[0] - filtered_cones[c][u][0]) ** 2 + (pt[1] - filtered_cones[c][u][1]) ** 2)
                    if distance > allowed_distance:
                        new_cone = True
                    else:
                        new_cone = False
                        break #it stops looking   
                     
            if new_cone == True:
                cone_number[c] += 1
                filtered_cones[c].append(pt)
                width_height[c].append([w, h])
                
    return frame, filtered_cones, width_height
                
"""  
        
        


#intersection over union
def compute_iou(old_cone_coordinates, old_width_height, cone_coordinates, width_height):
    box_old = [old_cone_coordinates[0], old_cone_coordinates[1], old_cone_coordinates[0] + old_width_height[0], old_cone_coordinates[1] + old_width_height[1]]
    box_new = [cone_coordinates[0], cone_coordinates[1], cone_coordinates[0] + width_height[0], cone_coordinates[1] + width_height[1]]
    
    # Calculate intersection area
    intersection_width = min(box_old[2], box_new[2]) - max(box_old[0], box_new[0])
    intersection_height = min(box_old[3], box_new[3]) - max(box_old[1], box_new[1])
    
    if intersection_width <= 0 or intersection_height <= 0:
        return 0
    
    intersection_area = intersection_width * intersection_height

    # Calculate union area
    box1_area = (box_old[2] - box_old[0]) * (box_old[3] - box_old[1])
    box2_area = (box_new[2] - box_new[0]) * (box_new[3] - box_new[1])
    
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def check_new_cones(cone_coordinates, width_height, old_cone_coordinates, old_width_height):
    for q in range(0, 2):
        for p in range (0, len(old_cone_coordinates[q])):
            old_cone_coordinates[q][p][2] += 1
    final_cone_coordinates = old_cone_coordinates.copy()
            
    new_width_height = old_width_height.copy()
    for i in range(0, 2):
        for k in range(0, len(cone_coordinates[i])):
            new_cone = True
            for j in range(0, len(old_cone_coordinates[i])):
                iou = compute_iou(old_cone_coordinates[i][j], old_width_height[i][j], cone_coordinates[i][k], width_height[i][k])
                if iou > 0.5:
                    new_cone = False
                    final_cone_coordinates[i][j] = cone_coordinates[i][k]
                    final_cone_coordinates[i][j][2] = 0
                    new_width_height[i][j] = width_height[i][k]
                    
            if new_cone == True:
                final_cone_coordinates[i].append(cone_coordinates[i][k])
                new_width_height[i].append(width_height[i][k])        
    old_cone_coordinates, old_width_height = final_cone_coordinates.copy(), new_width_height.copy()
    
    #removing cones that have not been found for x frame(s)
    temp_coordinates = old_cone_coordinates.copy()
    for i in range(0, 2):
        for j in range(len(temp_coordinates[i]) - 1, -1, -1):
            if temp_coordinates[i][j][2] > 3:  # X: this is the place to chance the number of frames a cone can be missing
                del old_cone_coordinates[i][j]
                del old_width_height[i][j]
    
                
    return old_cone_coordinates, old_width_height

def preprocess_image(frame, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std, templates):
    t1 = time.time()
    frame = color_transfer(frame, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std)
    frame_yellow = color_enhancement(frame)
    frame_blue = frame_yellow.copy()
    frame_yellow = find_yellow(frame_yellow)
    frame_blue = find_blue(frame_blue)
    frame = cv2.add(frame_yellow, frame_blue)            
    frame, cone_coordinates, width_height = template_matching(frame,templates)
    t2 = time.time()
    fps = 1/(t2-t1)
    #print(f"FPS = {fps}")
    return frame, cone_coordinates, width_height, fps

def draw_cones(frame, cone_coordinates, width_height, Object_tracking):
    #drawing the rectangles around the cones      
    for p in range (0, 2):
        next_img = 0
        if p == 1:
            color = (255, 0, 0)
        else:
            color = (0, 255, 255)
        if Object_tracking == True:
            for idx, pt in enumerate(cone_coordinates[p]):
                #put text on the cones´ rectangles
                cv2.putText(frame, str(idx), (pt[0], pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (pt[0], pt[1]), (pt[0] + width_height[p][next_img][0], pt[1] + width_height[p][next_img][1]), color, 2)
                next_img += 1
        else:
            for pt in cone_coordinates[p]:
                cv2.rectangle(frame, (pt[0], pt[1]), (pt[0] + width_height[p][next_img][0], pt[1] + width_height[p][next_img][1]), color, 2)
                next_img += 1

def ReadAnnotationFile(img, image_name, Testpath_labels):
    with open(Testpath_labels + image_name[:-4] + ".txt") as f:
        Cones = []
        for line in f:
            # Split the line into a list
            line = line.split()

            # Interpret color
            if line[0] == "0":
                color = "yellow"
            elif line[0] == "1":
                color = "blue"
            else:
                # Skip the current line if the color is not recognized
                continue
                
            x = int(float(line[1]) * img.shape[1])
            y = int(float(line[2]) * img.shape[0])
            w = int(float(line[3]) * img.shape[1])
            h = int(float(line[4]) * img.shape[0])

            # Extract the cone location and color from the list
            Cone_location = [(x,y), (w,h), color]  
            
            Cones.append(Cone_location)   

    return Cones 

def convert_array(cone_coordinates, width_height):
    #converting the coordinates to the same format as the ones from the annotation file
    con_pos_wh_color = []
    color = ["blue", "yellow"]
    for p in range(0, 2):
        for q in range(0, len(cone_coordinates[p])):
            cone=[cone_coordinates[p][q], width_height[p][q], color[p]]
            con_pos_wh_color.append(cone)
    
    return con_pos_wh_color      
         
def IOU(boxA, boxB):
    # Extract the coordinates of the boxes
    x0A, y0A, x1A, y1A = boxA
    x0B, y0B, x1B, y1B = boxB
    
    # Determine the (x, y)-coordinates of the intersection rectangle
    l_x = max(x0A, x0B)
    r_x = min(x1A, x1B)
    t_y = max(y0A, y0B)
    b_y = min(y1A, y1B)

    # Compute the area of intersection rectangle
    interArea = max(0, r_x - l_x) * max(0, b_y - t_y)
    
    # If the area is non-positive, the boxes don't intersect
    if interArea <= 0:
        Iou = 0
        return Iou

    # Compute the area of both rectangles
    area_box_a = abs(x1A - x0A) * abs(y1A - y0A)
    area_box_b = abs(x1B - x0B) * abs(y1B - y0B)

    # Compute the intersection over union
    Union = area_box_a + area_box_b - interArea

    if interArea > area_box_a + area_box_b:
        print("Error: wtf")

    # Compute the intersection over union
    Iou = interArea / Union

    #print("Iou: " + str(Iou))
    return Iou
         
# Test Logic
def test_logic(Testpath_images = "Hog/Test/images/", Testpath_labels = "Hog/Test/label/"): #Hog\Test\images\amz_01361.png #
    L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std = finds_LAB_reference_from_folder("processing_ZED//vores")
    templates = create_templates2()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    fps_metric = []

    # the first image in the test folder
    for images in os.listdir(Testpath_images):
        # Read the image
        
        
        img = cv2.imread(Testpath_images + images)
        # Read the Annotation file one line at a time
        
        Cones_from_ann = ReadAnnotationFile(img, images, Testpath_labels)

        _, cone_coordinates, width_height, fps = preprocess_image(img, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std, templates)
        
        fps_metric.append(fps)

        # Detect cones in the frame
        cone_locations_pre = convert_array(cone_coordinates, width_height)

        # Initiate the state of the cones as the lenght of the cones from the annotation file
        Close_state_ann = len(Cones_from_ann) * [False]
        close_state_hog = len(cone_locations_pre) * [False]

        # Run a intersection over union check to see if the cones are close to each other
        for cone in cone_locations_pre:
            close_cones = []
            for i, cone_from_ann in enumerate(Cones_from_ann):
                # Extract the coordinates of the boxes
                              
                # Extracting coordinates for cone A
                x0A = max(cone[0][0],0)
                y0A = max(cone[0][1], 0)
                x1A = max(cone[0][0]+cone[1][0], 0)
                y1A = max(cone[0][1]+cone[1][1], 0)

                # Extracting coordinates for cone B
                x0B = max(cone_from_ann[0][0] - cone_from_ann[1][0] // 2, 0)
                y0B = max(cone_from_ann[0][1] - cone_from_ann[1][1] // 2, 0)
                x1B = max(cone_from_ann[0][0] + cone_from_ann[1][0] // 2, 0)
                y1B = max(cone_from_ann[0][1] + cone_from_ann[1][1] // 2, 0)

                # Calculate the intersection over union
                Iou = IOU((x0A, y0A, x1A, y1A), (x0B, y0B, x1B, y1B))
                #print("Iou: " + str(Iou))

                if Iou >= 0.5:
                    # If the cones are close to each other, save the index, and the IOU value. Only the closest cone will be saved
                    close_cones.append((i, Iou))

            # If there are any close cones, save the closest one
            if close_cones:
                # Mark hog cone as found
                close_state_hog[cone_locations_pre.index(cone)] = True

                # Sort the list of close cones by IOU value
                close_cones.sort(key=lambda x: x[1], reverse=True)

                # Save the index of the closest cone
                Close_state_ann[close_cones[0][0]] = True        

        true_positives += close_state_hog.count(True)
        false_positives += close_state_hog.count(False)
        false_negatives += Close_state_ann.count(False)

    # We have chosen to set the precision to 0 if there are no true positives and no false positives as this is an undefinable case 
    if true_positives + false_positives + false_negatives == 0:
        Precision = 1
        Recall = 1
    elif true_positives + false_positives == 0:
        Precision = 0
    elif (true_positives + false_negatives) == 0:
        Recall = 0
    else:
        Recall = true_positives/ (true_positives + false_negatives)
        Precision = true_positives / (true_positives + false_positives)   

    print("Recall: " + str(Recall))
    print("Precision: " + str(Precision))
    print(fps_metric)
    print("Average FPS: " + str(sum(fps_metric)/len(fps_metric)))
    print("-------------------------------------------------")

    """
    # Draw the found cones with blue  
    for cone in Cones_from_ann:
        if Close_state_ann[Cones_from_ann.index(cone)]:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.rectangle(img, (cone[0][0] - cone[1][0]//2 , cone[0][1] - cone[1][1]//2), (cone[0][0] + cone[1][0]//2, cone[0][1] + cone[1][1]//2), color, 2)         

    # Draw all the cones found 
    for cone in cone_locations_pre:
        cv2.rectangle(img, (cone[0][0], cone[0][1]), (cone[0][0] + cone[1][0], cone[0][1] + cone[1][1]), (255, 0, 0), 2)

    
    # Display the frame - rezie the image to fit the screen
    img = cv2.resize(img, (1080, 720))

    cv2.imshow("Frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
    """  
            
Object_tracking = False

def load(Object_tracking):
    #load video from folder:
    video_folder = "processing_ZED//ZED_color_video_Run_1.avi"
    cap = cv2.VideoCapture(video_folder)
    L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std = finds_LAB_reference_from_folder("processing_ZED//vores")
    frame_number = 0
    
    while True:
        t1 = time.time()
        frame_number += 1
        # Read the frames of the video
        ret , frame = cap.read()    
        
        if ret == False:
            break
        
        frame, cone_coordinates, width_height, _ = preprocess_image(frame, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std)
        if Object_tracking == True:
            #makes sure that there are a counting entry for each cone
            for i in range(0, 2):
                    for j in range(0, len(cone_coordinates[i])):
                        cone_coordinates[i][j].append(0)
                        
            if frame_number == 1:
                old_cone_coordinates, old_width_height = cone_coordinates.copy(), width_height.copy()
            else:
                old_cone_coordinates, old_width_height = check_new_cones(cone_coordinates, width_height, old_cone_coordinates, old_width_height)        
            
            #draw_cones(frame, old_cone_coordinates, old_width_height, y, Object_tracking)
        #else:
            #draw_cones(frame, cone_coordinates, width_height, y, Object_tracking)

        #show the frames:
        #cv2.imshow("Video", frame)
        t2 = time.time()
        #print(f" FPS ={1/(t2-t1)}")

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
#load(Object_tracking)
test_logic()