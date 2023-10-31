#from PIL import Image
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

input_folder = "cone_tracking_data"
output_folder = "preprocessed_images"
folder = "Thor_data"

#calculate the mean value all of the images in folder:
def calculate_mean(RGB):        
    mean1 = 0
    mean_R = 0
    mean_G = 0
    mean_B = 0
    total_mean = [ 0, 0, 0 ]   #"B", "G", "R"
    numb_images = 0
    folder = "Thor_data"
    
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
    
def median_substraction(x):
    global processed_img, temp_img
    
    if processed_img.shape[2] == 3:
        RGB = True
    else:
        RGB = False
    
    mean_RGB = calculate_mean(RGB)
    # Process each pixel value
    for y in range(processed_img.shape[0]):
        for x in range(processed_img.shape[1]):
            # Get the RGB values for the current pixel
            pixel = processed_img[y, x]

            # Subtract the mean values from each channel
            pixel[0] -= mean_RGB[2] #B
            pixel[1] -= mean_RGB[1] #G
            pixel[2] -= mean_RGB[0] #R
            temp_img[y, x] = pixel


        

def preprocessing():
    input_folder = "cone_tracking_data"
    output_folder = "preprocessed_images"
    folder = "Thor_data"
                
    mean_RGB = calculate_mean(folder, RGB = True)
    print("Preprocessing started.")
            
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(input_folder, filename))
            
            #median substraction
            # Process each pixel value
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    # Get the RGB values for the current pixel
                    pixel = image[y, x]

                    # Subtract the mean values from each channel
                    pixel[0] -= mean_RGB[2] #B
                    pixel[1] -= mean_RGB[1] #G
                    pixel[2] -= mean_RGB[0] #R
                    image[y, x] = pixel
            cv2.imshow("median substraction", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
            #image sharpening 
            #eventuelt blur først
            #image smoothing
            #Normalize color/brightness
            #noise removal
            
            
            
            
            
            
            # good link for preprocessing
            #https://www.analyticsvidhya.com/blog/2023/03/getting-started-with-image-processing-using-opencv/
        
        
            # Save the preprocessed image
            image.save(os.path.join(output_folder, filename))

#converter til float int point inden databehandling. hint fra andreas.
#trying to remove contrast in image 
img = cv2.imread("Images//FrameRemoved//Image_8.png")
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

