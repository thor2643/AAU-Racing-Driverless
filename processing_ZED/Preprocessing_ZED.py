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

def remove_all_but_concrete( img1):
    img = img1.copy()
    lower_HSV = [0, 0, 72]
    upper_HSV= [200, 80, 100]
    lower_HSV = np.array(lower_HSV, dtype=np.uint8)  # Convert to NumPy array
    upper_HSV = np.array(upper_HSV, dtype=np.uint8)  # 1Convert to NumPy array

    # 1. Convert the image to HSV color space
    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Apply HSV thresholding to find concrete areas
    HSV_mask = cv2.inRange(HSV_img, lower_HSV, upper_HSV)
    
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

def create_templates():
    frame = cv2.imread("preproccesing//frame.jpg")
    #frame = cv2.imread("preproccesing//frame1 15 .jpg")
    #cutting both templates out of the frame
    blue_template = frame[270:325, 35:72]
    yellow_template = frame[268:305, 560:586]
    
    
    
    #make the templates smaller to matcg the size of the cones
    yellow_template1 = cv2.resize(yellow_template, (int(26 * 1.25), int(37 * 1.25)))
    blue_template1 = cv2.resize(blue_template, (int(37 * 0.6), int(55 * 0.6)))
    yellow_template2 = cv2.resize(yellow_template, (int(26 * 0.6), int(37 * 0.6)))
    blue_template2 = cv2.resize(blue_template, (int(22 * 0.6), int(33 * 0.6)))
    yellow_template3 = cv2.resize(yellow_template, (int(15 * 0.6), int(22 * 0.6)))
    blue_template3 = cv2.resize(blue_template, (int(13 * 0.6), int(19 * 0.6)))
    blue_template = cv2.resize(blue_template, (int(37 * 0.80), int(55 * 0.8)))
    
    

    # Save image to directory
    #cv2.imwrite("yellow_template.jpg", yellow_template)
    cv2.imwrite("blue_template.jpg", blue_template)
    # cv2.imwrite("yellow_template.jpg", yellow_template)
    #cv2.imwrite("yellow_template1.jpg", yellow_template1)
    #cv2.imwrite("blue_template1.jpg", blue_template1)
    #cv2.imwrite("yellow_template2.jpg", yellow_template2)
    #cv2.imwrite("blue_template2.jpg", blue_template2)
    #cv2.imwrite("yellow_template3.jpg", yellow_template3)
    #cv2.imwrite("blue_template3.jpg", blue_template3)

def template_matching(frame, y):
    #reading all the templates
    #yellow_template = cv2.imread("preproccesing//preprocesing_img//yellow_template.jpg")
    blue_template = cv2.imread("preproccesing//preprocesing_img//blue_template.jpg")
    yellow_template1 = cv2.imread("preproccesing//preprocesing_img//yellow_template1.jpg")
    blue_template1 = cv2.imread("preproccesing//preprocesing_img//blue_template1.jpg")
    yellow_template2 = cv2.imread("preproccesing//preprocesing_img//yellow_template2.jpg")
    blue_template2 = cv2.imread("preproccesing//preprocesing_img//blue_template2.jpg")
    yellow_template3 = cv2.imread("preproccesing//preprocesing_img//yellow_template3.jpg")
    #blue_template3 = cv2.imread("preproccesing//preprocesing_img//blue_template3.jpg")
    
    #Variables
    c = 0
    cone_number = [(0),(0)]
    allowed_distance=20   #pixels
    new_cone = True
    distance=0
    filtered_cones= [[], []]
    width_height = [[],[]]
    i = 0
    threshold = 0.6
    
    #size of the frame
    frame_height, frame_width = frame.shape[:2]
    #resizing the frame to make cumputations faster
    frame_copy = frame[y : frame_height , 0 : frame_width]
    #yellow_template blue_template3
    template = [yellow_template1, yellow_template2, yellow_template3, blue_template ,blue_template1, blue_template2]
    
    for i, template in enumerate(template):
        if i == 3:
            c = 1
            new_cone = True
            threshold = 0.65
                
        w, h = template.shape[1], template.shape[0]
        res = cv2.matchTemplate(frame_copy,template,cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)
        
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
                filtered_cones[c].append(list(pt))    
                width_height[c].append([w, h])
        
    return frame, filtered_cones, width_height

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

def preprocess_image(frame, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std):
    frame = color_transfer(frame, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std)
    frame_yellow = color_enhancement(frame)
    y = remove_all_but_concrete(frame_yellow)
    frame_blue = frame_yellow.copy()
    frame_yellow = find_yellow(frame_yellow)
    frame_blue = find_blue(frame_blue)
    frame = cv2.add(frame_yellow, frame_blue)            
    frame, cone_coordinates, width_height = template_matching(frame, y)
    return frame, cone_coordinates, width_height, y

def draw_cones(frame, cone_coordinates, width_height, y, Object_tracking):
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
                cv2.putText(frame, str(idx), (pt[0], pt[1] + y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (pt[0], pt[1] + y), (pt[0] + width_height[p][next_img][0], pt[1] + width_height[p][next_img][1] + y), color, 2)
                next_img += 1
        else:
            for pt in cone_coordinates[p]:
                cv2.rectangle(frame, (pt[0], pt[1] + y), (pt[0] + width_height[p][next_img][0], pt[1] + width_height[p][next_img][1] + y), color, 2)
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
    cone = []
    color = ["blue", "yellow"]
    for p in range(0, 2):
        for q in range(0, len(cone_coordinates[p])):
            cone.append((cone_coordinates[p][q], width_height[p][q], color[p])) 
            con_pos_wh_color.append(cone)
    
    return con_pos_wh_color      
         
def IOU(boxA, boxB):
    # Extract the coordinates of the boxes
    x0A, y0A, x1A, y1A = boxA
    x0B, y0B, x1B, y1B = boxB
    
    # Determine the (x, y)-coordinates of the intersection rectangle
    r_x = max(x0A, x0B)
    t_y = max(y0A, y0B)
    l_x = min(x1A, x1B)
    b_y = min(y1A, y1B)

    # Compute the area of intersection rectangle
    interArea = max(0, l_x - r_x) * max(0, t_y - b_y )
    
    # If the area is non-positive, the boxes don't intersect
    if interArea <= 0:
        Iou = 0
        return Iou

    # Compute the area of both rectangles
    area_box_a = (x1A - x0A) * (y1A - y0A)
    area_box_b = (x1B - x0B) * (y1B - y0B)

    # Compute the intersection over union
    Union = area_box_a + area_box_b - interArea

    # Compute the intersection over union
    Iou = interArea / Union

    print("Iou: " + str(Iou))
    return Iou
         
         
# Test Logic
def test_logic(Testpath_images = "AAU-RACING-DRIVERLESS/Hog/Test/images/", Testpath_labels = "Hog\Test\label"):
    L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std = finds_LAB_reference_from_folder("processing_ZED//vores")

    # the first image in the test folder
    for images in os.listdir(Testpath_images):
        # Read the image
        img = cv2.imread(Testpath_images + images)
        # Read the Annotation file one line at a time

        Cones_from_ann = ReadAnnotationFile(img, images, Testpath_labels)

        _, cone_coordinates, width_height, _ = preprocess_image(img, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std)

        # Detect cones in the frame
        cone_locations_HOG = convert_array(cone_coordinates, width_height)

        # Initiate the state of the cones as the lenght of the cones from the annotation file
        Close_state_ann = len(Cones_from_ann) * [False]
        close_state_hog = len(cone_locations_HOG) * [False]

        # Run a intersection over union check to see if the cones are close to each other - THIS IS OLD CODE
        for cone in cone_locations_HOG:
            close_cones = []
            for i, cone_from_ann in enumerate(Cones_from_ann):
                # Extract the coordinates of the boxes

                # Extracting coordinates for cone A
                x0A = max(cone[2][0] - cone[4][0] // 2, 0)
                y0A = max(cone[2][1] + cone[4][1] // 2, 0)
                x1A = max(cone[2][0] + cone[4][0] // 2, 0)
                y1A = max(cone[2][1] - cone[4][1] // 2, 0)

                # Extracting coordinates for cone B
                x0B = max(cone_from_ann[0][0] - cone_from_ann[1][0] // 2, 0)
                y0B = max(cone_from_ann[0][1] + cone_from_ann[1][1] // 2, 0)
                x1B = max(cone_from_ann[0][0] + cone_from_ann[1][0] // 2, 0)
                y1B = max(cone_from_ann[0][1] - cone_from_ann[1][1] // 2, 0)

                # Calculate the intersection over union
                Iou = IOU((x0A, y0A, x1A, y1A), (x0B, y0B, x1B, y1B))

                if Iou >= 0.5:
                    # If the cones are close to each other, save the index, and the IOU value. Only the closest cone will be saved
                    close_cones.append((i, Iou))

            # If there are any close cones, save the closest one
            if close_cones:
                # Mark hog cone as found
                close_state_hog[cone_locations_HOG.index(cone)] = True

                # Sort the list of close cones by IOU value
                close_cones.sort(key=lambda x: x[1], reverse=True)

                # Save the index of the closest cone
                Close_state_ann[close_cones[0][0]] = True        

        true_positives = close_state_hog.count(True)
        false_positives = close_state_hog.count(False)
        false_negatives = Close_state_ann.count(False)

        # We have chosen to set the precision to 0 if there are no true positives and no false positives as this is an undefinable case 
        if true_positives + false_positives == 0:
            Precision = 0
        elif (true_positives + false_negatives) == 0:
            Recall = 0
        else:
            Recall = true_positives/ (true_positives + false_negatives)
            Precision = true_positives / (true_positives + false_positives)   

        print(Recall)

        print("Recall: " + str(Recall))
        print("Precision: " + str(Precision))

        # Draw the found cones with blue  
        for cone in Cones_from_ann:
            if Close_state_ann[Cones_from_ann.index(cone)]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(img, (cone[0][0] - cone[1][0]//2 , cone[0][1] - cone[1][1]//2), (cone[0][0] + cone[1][0]//2, cone[0][1] + cone[1][1]//2), color, 2)         

        # Draw all the cones found 
        for cone in cone_locations_HOG:
            cv2.rectangle(img, (cone[2][0] - cone[4][0]//2, cone[2][1] - cone[4][1]//2), (cone[2][0] + cone[4][0]//2, cone[2][1] + cone[4][1]//2), (255, 0, 0), 2)

        # Display the frame - rezie the image to fit the screen
        img = cv2.resize(img, (1080, 720))

        cv2.imshow("Frame", img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break     
            
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
        
        frame, cone_coordinates, width_height, y = preprocess_image(frame, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std)
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
        print(f" FPS ={1/(t2-t1)}")
        print(cone_coordinates)
        print(width_height)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
#load(Object_tracking)
test_logic()