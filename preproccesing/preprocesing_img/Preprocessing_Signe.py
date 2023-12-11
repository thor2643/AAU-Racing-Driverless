#from PIL import Image
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import time
import matplotlib.pyplot as plt

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
    lower_HSV = [0, 0, 55]
    upper_HSV= [117, 77, 100]
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
    lower_yellow = np.array([20, 100, 0]) 
    upper_yellow = np.array([40, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    temp_img = cv2.bitwise_and(processed_img, processed_img, mask=mask) 
    return temp_img
    
def find_blue(processed_img):

    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV 
    lower_blue = np.array([100, 50, 50])
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

def template_matching(frame, y,):
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
        
        #going throug each found location and drawing a rectangle around it,
        # if it is not too close to another cone
        for pt in zip(*loc[::-1]):
            #sort out the ones that are too close to each other
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

def find_center_cords(cone_cordinates, width_height):
    #running through all the cone coordinates and finding the center coordinates
    centercordinates = [[],[]]
    for p in range (0, 2):
        next_img = 0
        for pt in cone_cordinates[p]:
            centercordinates[p].append([pt[0] + width_height[p][next_img][0] / 2, pt[1] + width_height[p][next_img][1] / 2])
            next_img += 1
    return centercordinates

def find_polynomial_coefficients(points, degree):
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])

    # Fit a polynomial of the specified degree
    coefficients = np.polyfit(x, y, degree)

    return coefficients

def plot_polynomial(coefficients, x_range, label):
    y_range = np.polyval(coefficients, x_range)
    plt.plot(x_range, y_range, label=label)

def draw_line_between_cones(title, coordinates, x_range, y_range,):
    # Extract x and y values
    x_values = [coord[0] for coord in coordinates]
    y_values = [coord[1] for coord in coordinates]

    # Fit a quadratic polynomial
    degree = 1  # Adjust the degree as needed
    poly_coefficients = find_polynomial_coefficients(coordinates, degree)

    # Plot the original points
    plt.scatter(x_values, y_values, label='Data Points')
    x_interval = np.linspace(0, x_range[1], 5000)

    # Plot the polynomial curve
    plot_polynomial(poly_coefficients, x_interval, f'Quadratic Regression (Degree {degree})')

    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # Set x and y range if specified
    if x_range is not None:
        plt.xlim(min(x_range), max(x_range))
    if y_range is not None:
        plt.ylim(min(y_range), max(y_range))

    plt.title(title)
    # Invert the y-axis
    plt.gca().invert_yaxis()
    plt.show()

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

def draw_cones(frame, cone_coordinates, width_height, y):
    #drawing the rectangles around the cones      
    for p in range (0, 2):
        next_img = 0
        if p == 1:
            color = (255, 0, 0)
        else:
            color = (0, 255, 255)
        for idx, pt in enumerate(cone_coordinates[p]):
            #put text on the cones´ rectangles
            cv2.putText(frame, str(idx), (pt[0], pt[1] + y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (pt[0], pt[1] + y), (pt[0] + width_height[p][next_img][0], pt[1] + width_height[p][next_img][1] + y), color, 2)
            next_img += 1

def load():
    #load video from folder:
    video_folder = "preproccesing//ZED_color_video_Run_1.avi" #preproccesing//ZED_color_video_Run_1.avi Data_AccelerationTrack//1//Color.avi
    cap = cv2.VideoCapture(video_folder)
    L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std = finds_LAB_reference_from_folder("preproccesing//test//FullDataset//images//vores")  #Images//Color_transfer #
    frame_number = 0
    
    while True:
        t1 = time.time()
        frame_number += 1
        # Read the frames of the video
        ret , frame = cap.read()    
        
        if ret == False:
            break
        
        frame, cone_coordinates, width_height, y = preprocess_image(frame, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std)
        #makes sure that there are a counting entry for each cone
        for i in range(0, 2):
                for j in range(0, len(cone_coordinates[i])):
                    cone_coordinates[i][j].append(0)
                    
        if frame_number == 1:
            old_cone_coordinates, old_width_height = cone_coordinates.copy(), width_height.copy()
        else:
            old_cone_coordinates, old_width_height = check_new_cones(cone_coordinates, width_height, old_cone_coordinates, old_width_height)        
        
        draw_cones(frame, old_cone_coordinates, old_width_height, y)
        #show the frames:
        cv2.imshow("Video", frame)
        frame_number += 1
        t2 = time.time()
        print(t2-t1)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
load()
  

        #size_x_frame = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        #centercordinates = find_center_cords(cone_cordinates, width_height)
        #draw_line_between_cones("Yellow cones", centercordinates[0], x_range=[0, 640], y_range=[0, 480])
        #draw_line_between_cones("Blue cones", centercordinates[1], x_range=[0, 640], y_range=[0, 480])