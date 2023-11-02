#from PIL import Image
import os
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance 

#-------------------------------------------------------------
def find_blue(x):
    global processed_img, temp_img

    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV 
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    temp_img = cv2.bitwise_and(processed_img, processed_img, mask=mask)  
    
    
    
find_blue()

#-------------------------------------------------------------
def find_yellow(x):
    global processed_img, temp_img

    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)

    # Define range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    temp_img = cv2.bitwise_and(processed_img, processed_img, mask=mask)  



#-------------------------------------------------------------

def color_enhancement(x):
    global processed_img, temp_img
    # Convert BGR to RGB colorspace
    image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    im = Image.fromarray(image_rgb)

    # Creating an object of Color class
    im3 = ImageEnhance.Color(im)

    # this factor controls the enhancement factor. 0 gives a black and white image. 1 gives the original image
    enhanced_image = im3.enhance(2.0)
    
    # Convert the enhanced image to an OpenCV format
    temp_img = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)


color_enhancement()




def color_enhancement_hsv():
    #global processed_img, temp_img
    processed_img = cv2.imread("Images\FrameRemoved\Image_6.jpg")
    #enhance the color of the processed_img
    #convert the image to HSV
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    #split the image into channels
    h, s, v = cv2.split(hsv)
    #enhance the saturation channel
    s = cv2.equalizeHist(s)
    #merge the channels together
    hsv = cv2.merge([h, s, v])
    #convert the image back to BGR
    temp_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    cv2.imshow('Original Image', processed_img)
    cv2.imshow('Enhanced Image', temp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

#-------------------------------------------------------------
def noise_removal_median():
    global processed_img, temp_img
    temp_img = cv2.medianBlur(processed_img, 3)


#-------------------------------------------------------------
def noise_removal_mean():
    global processed_img, temp_img
    kernel = np.array([[1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9]])
    temp_img = cv2.filter2D(processed_img, -1, kernel)
#-------------------------------------------------------------

def sharpening():
    global processed_img, temp_img
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    temp_img = cv2.filter2D(processed_img, -1, kernel)
#-------------------------------------------------------------------------------
def laplacian_sharpening():
    #global processed_img, temp_img
    img = cv2.imread("Images\FrameRemoved\Image_6.jpg")
    # Load image
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply filter
    img_filter = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    # Convert back to 8-bit image
    img_filter = cv2.convertScaleAbs(img_filter)
    # Display image
    cv2.imshow('High-pass Filtered Image', img_filter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-------------------------------------------------------------
#color transfer - based on the images in folder:

def LAB_for_reference(folder):
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
    

#color transfer
#change the color of the processed_img to match the color of the reference image
def color_transfer(procesed_img):
    procesed_img = cv2.imread("Images\FrameRemoved\Image_6.jpg")
    #the folder where the reference images are located
    folder = "Images\FrameRemoved"

    #convert the images from RGB to LAB
    img_target = cv2.cvtColor(procesed_img, cv2.COLOR_BGR2LAB).astype("float32")
    #splitting the images into channels
    L_t ,A_t ,B_t =cv2.split(img_target)
    
    #computing the mean and standard deviation of each channel for both images
    #for source image
    L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std = LAB_for_reference(folder)
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
    transfer_img = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    
#--------------------------------------------------------------------------------------
    #color transfer
#change the color of the processed_img to match the color of the reference image
def color_transfer(x):
    global processed_img, temp_img
    reference_img = cv2.imread("Images\FrameRemoved\Image_3.jpg")

    #convert the images from RGB to LAB
    img_source = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB).astype("float32")
    img_target = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB).astype("float32")
    #splitting the images into channels
    L_s ,A_s ,B_s =cv2.split(img_source)
    L_t ,A_t ,B_t =cv2.split(img_target)
    
    #computing the mean and standard deviation of each channel for both images
    #for source image
    L_s_mean, L_s_std = L_s.mean(), L_s.std()
    A_s_mean, A_s_std = A_s.mean(), A_s.std()
    B_s_mean, B_s_std = B_s.mean(), B_s.std()
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
    
    
#-------------------------------------------------------------
#calculate the mean value all of the images in folder:
def calculate_mean(RGB):        
    mean1 = 0
    mean_R = 0
    mean_G = 0
    mean_B = 0
    total_mean = [ 0, 0, 0 ]   #"B", "G", "R"
    numb_images = 0
    folder = "Images\FrameRemoved"
    
    for filename in os.listdir(folder):
        if filename.endswith(".jpg" or ".png"):
            image = cv2.imread(os.path.join(folder, filename))
            if RGB==True:
                B, G, R = cv2.split(image)
    
                # Sum the channel values
                mean_R += np.mean(R)
                mean_G += np.mean(G)
                mean_B += np.mean(B)
                numb_images += 1   
            else:
                mean = np.mean(image)
                mean1 += mean
                numb_images += 1  
    
    if RGB==True:
        mean_RGB=[mean_B, mean_G, mean_R]
        for i in range(0, 3):
            total_mean[i] = mean_RGB[i]/numb_images
    else: 
        total_mean = mean1/numb_images
    return total_mean   
    
def mean1_substraction(x):
    global processed_img, temp_img
    
    print (f" temp shape: {temp_img.shape}")
    if len(temp_img.shape) == 3:
        RGB = True
    else:
        RGB = False
    
    mean_RGB = calculate_mean(RGB)
    
    if RGB == True:
        # Process each pixel value
        for y in range(temp_img.shape[0]):
            for x in range(temp_img.shape[1]):
                # Get the RGB values for the current pixel
                pixel = temp_img[y, x]

                # Subtract the mean values from each channel
                pixel[0] -= mean_RGB[2] #B
                pixel[1] -= mean_RGB[1] #G
                pixel[2] -= mean_RGB[0] #R
                temp_img[y, x] = pixel
    else:
        temp_img = cv2.subtract(temp_img, mean_RGB)


#Forsøg ----------------------------------------------------
#Hough Line Transform
# dst = cv2.Canny(img, 50, 200, None, 3)
# lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
# # Draw the lines
# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#         cv2.line(dst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
# cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
# plt.imshow(cdst)
# cv2.imshow("detected lines", cdst)
# cv2.waitKey(0)  
# cv2.destroyAllWindows()

          
#--------------------------------------------------------------------------------------------------
 
        #image sharpening 
        #eventuelt blur først
        #image smoothing
        #Normalize color/brightness
        #noise removal
        
        
        
        
        
        
        # good link for preprocessing
        #https://www.analyticsvidhya.com/blog/2023/03/getting-started-with-image-processing-using-opencv/
        
        


#converter til float int point inden databehandling. hint fra andreas.
#trying to remove contrast in image 
#img = cv2.imread("Images//FrameRemoved//Image_8.png")
# image_float = img.astype(np.float32)
# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# filtersize = 201 
# gaussianImg = cv2.GaussianBlur(grayImg, (filtersize, filtersize), 128)
# cv2.imshow('Converted Image', gaussianImg)
# cv2.waitKey(0)
# newImg = (grayImg - gaussianImg)

# cv2.imshow('New Image', newImg)
# cv2.imshow('Original Image', img) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#doing histogram equalization
# img = cv2.imread("Images//FrameRemoved//Image_8.png")
# plt.hist(img.ravel(),256,[0,256]) 

# plt.show() 
# #plt.savefig('hist.png')

# equ = cv2.equalizeHist(img)
# res = np.hstack((img,equ))

# cv2.imshow('Equalized Image',res)
# cv2.imwrite('Equalized Image.png',res) 

# plt.hist(res.ravel(),256,[0,256]) 

# plt.show() 
# #plt.savefig('equal-hist.png')
# cv2.imshow('Original Image', img)   
# cv2.waitKey(0)
# cv2.destroyAllWindows()

