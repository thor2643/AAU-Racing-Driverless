#from PIL import Image
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
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
    lower_yuv = [64, 113, 116]
    upper_yuv = [100, 138, 153]
    lower_yuv = np.array(lower_yuv, dtype=np.uint8)  # Convert to NumPy array
    upper_yuv = np.array(upper_yuv, dtype=np.uint8)  # 1Convert to NumPy array

    # 1. Convert the image to YUV color space
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 2. Apply YUV thresholding to find concrete areas
    yuv_mask = cv2.inRange(yuv_img, lower_yuv, upper_yuv)

    # 3. Find contours in the binary mask
    contours, _ = cv2.findContours(yuv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Initialize variables to keep track of the largest blob and its bounding box
    largest_blob = None
    largest_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_blob = contour

    if largest_blob is not None:
        # 5. Get the bounding box of the largest blob
        _ , y, _, _ = cv2.boundingRect(largest_blob)

        # 6. Crop the original image using the bounding box
        #concrete_area = img[y-5:y+5+h, x:x+w]
        
        return y

    # 7. If no concrete is found, return None
    return None

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
    im = Image.fromarray(image_rgb)
    # Creating an object of Color class
    im3 = ImageEnhance.Color(im)
    # this factor controls the enhancement factor. 0 gives a black and white image. 1 gives the original image
    enhanced_image = im3.enhance(2)
    # Convert the enhanced image to an OpenCV format
    temp_img = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
    return temp_img

def find_yellow(processed_img):

    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)

    # Define range of yellow color in HSV
    lower_yellow = np.array([20, 80, 80])  # 20, 80, 80 to get more yellow but also more noise
    upper_yellow = np.array([30, 255, 255])

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

def template_matching(frame, y):
    #reading all the templates
    yellow_template1 = cv2.imread("preproccesing//yellow_template.jpg")
    blue_template = cv2.imread("preproccesing//blue_template.jpg")
    yellow_template = cv2.imread("preproccesing//yellow_template1.jpg")
    blue_template1 = cv2.imread("preproccesing//blue_template1.jpg")
    yellow_template2 = cv2.imread("preproccesing//yellow_template2.jpg")
    blue_template2 = cv2.imread("preproccesing//blue_template2.jpg")
    yellow_template3 = cv2.imread("preproccesing//yellow_template3.jpg")
    blue_template3 = cv2.imread("preproccesing//blue_template3.jpg")
    
    #Variables
    c = 0
    cone_number = [(0),(0)]
    allowed_distance=30   #pixels
    new_cone = True
    distance=0
    filtered_cones= [[], []]
    width_height = [[],[]]
    i = 0
    threshold = 0.6
    
    #resizing the frame to make cumputations faster
    frame_copy = frame[y : 480 , 0 : 640]
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
                width_height[c].append([w, h])
                filtered_cones[c].append(pt)    
    
    #drawing the rectangles around the cones      
    for p in range (0, 2):
        next_img = 0
        if p == 1:
            color = (255, 0, 0)
        else:
            color = (0, 255, 255)
        for pt in filtered_cones[p]:
            cv2.rectangle(frame, (pt[0], pt[1] + y), (pt[0] + width_height[p][next_img][0], pt[1] + width_height[p][next_img][1] + y), color, 2)
            next_img += 1
    
    return frame, filtered_cones

def load():
    #load video from folder:
    video_folder = "Data_AccelerationTrack//1//Color.avi"
    cap = cv2.VideoCapture(video_folder)
    L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std = finds_LAB_reference_from_folder("Images//Color_transfer")
    time1 = 0
    
    while True:
        # Read the frames of the video
        _ , frame = cap.read()    
        
        if  time1 == 0 or time.time() - time1 > 0.18:  
            #process the frames:
            frame = color_transfer(frame, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std)
            frame_yellow = color_enhancement(frame)
            y = remove_all_but_concrete(frame)
            frame_blue = frame_yellow.copy()
            frame_yellow = find_yellow(frame_yellow)
            frame_blue = find_blue(frame_blue)
            frame = cv2.add(frame_yellow, frame_blue)
            frame, cone_cordinates = template_matching(frame, y)
            time1 =time.time()
            
            #show the frames:
            cv2.imshow("Video", frame)
    
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


load()