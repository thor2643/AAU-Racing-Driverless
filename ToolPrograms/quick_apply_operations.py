import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

def nothing(x): #dummy fuction
    pass

def hsv_tresh(x):
    global processed_img, temp_img

    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", 'HSV-color thresholds') #får fat i værdien af vores trackbars
    l_s = cv2.getTrackbarPos("LS", 'HSV-color thresholds')
    l_v = cv2.getTrackbarPos("LV", 'HSV-color thresholds')

    u_h = cv2.getTrackbarPos("UH", 'HSV-color thresholds')
    u_s = cv2.getTrackbarPos("US", 'HSV-color thresholds')
    u_v = cv2.getTrackbarPos("UV", 'HSV-color thresholds')

    l_b = np.array([l_h, l_s, l_v]) #nederst grænse for blå farve
    u_b = np.array([u_h, u_s, u_v]) #øverste grænse

    temp_img = cv2.inRange(hsv, l_b, u_b) #finder ud om hsv billede har dele som ligger i intervallet angivet af l_b og u_b

    #res = cv2.bitwise_and(img, img, mask=mask)

def convert_to_grey(x):
    global processed_img, temp_img

    temp_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

def sobel(x):
    global processed_img, temp_img

    if x == 1:
        temp_img = cv2.Sobel(processed_img, cv2.CV_32F, 1, 0, ksize=3)
    elif x == 2:
        temp_img = cv2.Sobel(processed_img, cv2.CV_32F, 0, 1, ksize=3)
    elif x == 3:
        grad_x = cv2.Sobel(processed_img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(processed_img, cv2.CV_32F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        temp_img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)    
    

def laplacian(x):
    global processed_img, temp_img

    dst = cv2.Laplacian(processed_img, cv2.CV_16S, ksize=3)

    temp_img = cv2.convertScaleAbs(dst)

def erode(x):
    global processed_img, temp_img

    if x%2 == 1:
        temp_img = cv2.erode(processed_img, np.ones((x,x), np.uint8), iterations=1)

def dilate(x):
    global processed_img, temp_img

    if x%2 == 1:
        temp_img = cv2.dilate(processed_img, np.ones((x,x), np.uint8), iterations=1)

def threshold(x):
    global processed_img, temp_img

    _, temp_img = cv2.threshold(processed_img, x, 255, cv2.THRESH_BINARY)

def apply_operation(x):
    global processed_img, temp_img

    processed_img = temp_img

    cv2.setTrackbarPos("Apply", "Image Operations", 0)
    cv2.setTrackbarPos("Apply", "HSV-color thresholds", 0)


def equalise_histogram(x):
    global processed_img, temp_img

    # convert it to grayscale
    img_yuv = cv2.cvtColor(processed_img,cv2.COLOR_BGR2YUV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

    #convert back
    temp_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
def historgram_stretching(x):
    global processed_img, temp_img
    # Apply histogram stretching
    min_intensity = np.min(processed_img)
    max_intensity = np.max(processed_img)

    # Linearly scale the pixel values to cover the full range [0, 255]
    stretched_image = cv2.convertScaleAbs(processed_img, alpha=255 / (max_intensity - min_intensity), beta=-255 * min_intensity / (max_intensity - min_intensity))
    temp = stretched_image



def mean_subtraction(x):
    global processed_img, temp_img

    # convert it to grayscale
    img_yuv = cv2.cvtColor(processed_img,cv2.COLOR_BGR2YUV)
    #img_yuv = cv2.cvtColor(processed_img,cv2.COLOR_BGR2HSV)

    #Gaussian filter
    mean_filter = 1/16 * np.array([[1,2,1],
                                    [2,4,2],
                                    [1,2,1]])

    
    filtersize = 513
    gaussianImg = cv2.GaussianBlur(img_HSV[:,:,2], (filtersize, filtersize), 128)
    #img_avg = cv2.filter2D(img_yuv[:,:,0],-1, mean_filter)

    img_HSV[:,:,2] = (img_HSV[:,:,2] - gaussianImg) #=cv2.subtract(img_gray, gaussianImg) #img_avg)


    temp_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    #just experimenting wit histogram equalisation vs stretching
        # convert it to grayscale
    img_yuv = cv2.cvtColor(temp_img,cv2.COLOR_BGR2YUV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])


    temp_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    #just experimenting wit histogram equalisation vs stretching
        # convert it to grayscale
    img_yuv = cv2.cvtColor(temp_img,cv2.COLOR_BGR2YUV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

    #convert back
    processed_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
                                #signe_historgram_stretching(temp_img)
    
    
    
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
    


#color transfer
#change the color of the processed_img to match the color of the reference image
def color_transfer(x):
    global processed_img, temp_img
    #the folder where the reference images are located
    folder = "Images\FrameRemoved"

    #convert the images from RGB to LAB
    img_target = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB).astype("float32")
    #splitting the images into channels
    L_t ,A_t ,B_t =cv2.split(img_target)
    
    #computing the mean and standard deviation of each channel for both images
    #for source image
    L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std = finds_LAB_reference_from_folder(folder)
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

def laplacian_sharpening(x):
    global processed_img, temp_img
    # Convert to grayscale
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    # Apply filter
    img_filter = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    # Convert back to 8-bit image
    temp_img = cv2.convertScaleAbs(img_filter)
    
def sharpening(x):
    global processed_img, temp_img
    #kernel can be changed to sharpen more or less
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    temp_img = cv2.filter2D(processed_img, -1, kernel)
    
def noise_removal_mean(x):
    global processed_img, temp_img
    kernel = np.array([[1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9]])
    temp_img = cv2.filter2D(processed_img, -1, kernel)

def noise_removal_median(x):
    global processed_img, temp_img
    temp_img = cv2.medianBlur(processed_img, 3)
    
def color_enhancement(x):
    global processed_img, temp_img
    # Convert BGR to RGB colorspace
    image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    # Convert to PIL image
    im = Image.fromarray(image_rgb)
    # Creating an object of Color class
    im3 = ImageEnhance.Color(im)
    # this factor controls the enhancement factor. 0 gives a black and white image. 1 gives the original image
    enhanced_image = im3.enhance(x)
    # Convert the enhanced image to an OpenCV format
    temp_img = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
    
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

def restart(x):
    global img, processed_img, temp_img, trackbars

    for window in trackbars:
        for trackbar_name in window[1]:
            cv2.setTrackbarPos(trackbar_name, window[0][0], 0)

    processed_img = img.copy()
    temp_img = img.copy()


def save_img(x):
    global processed_img, output_path, name_of_img

    path = output_path + "\\" + name_of_img + ".jpg"

    cv2.imwrite(path, processed_img)


# Add a "Save" button to the window
def save_values(x):
    if x == 1:
        values = [cv2.getTrackbarPos("LH", 'HSV-color thresholds'),
                  cv2.getTrackbarPos("LS", 'HSV-color thresholds'),
                  cv2.getTrackbarPos("LV", 'HSV-color thresholds'),
                  cv2.getTrackbarPos("UH", 'HSV-color thresholds'),
                  cv2.getTrackbarPos("US", 'HSV-color thresholds'),
                  cv2.getTrackbarPos("UV", 'HSV-color thresholds')]
        print("Values saved:", values)

#cap = cv2.VideoCapture(0)

cv2.namedWindow('HSV-color thresholds')
cv2.resizeWindow("HSV-color thresholds", 300, 350) 
cv2.createTrackbar("LH", "HSV-color thresholds", 0, 255, hsv_tresh) #LH = lower Hue
cv2.createTrackbar("LS", "HSV-color thresholds", 0, 255, hsv_tresh) #LS = lower saturation
cv2.createTrackbar("LV", "HSV-color thresholds", 0, 255, hsv_tresh) #Value
cv2.createTrackbar("UH", "HSV-color thresholds", 255, 255, hsv_tresh) #U =  upper
cv2.createTrackbar("US", "HSV-color thresholds", 255, 255, hsv_tresh)
cv2.createTrackbar("UV", "HSV-color thresholds", 255, 255, hsv_tresh)

cv2.createTrackbar("Apply", "HSV-color thresholds", 0, 1, apply_operation)
cv2.createTrackbar("Save", "HSV-color thresholds", 0, 1, save_values)


cv2.namedWindow('Image Operations')
cv2.resizeWindow("Image Operations", 400, 375) 
cv2.createTrackbar("Greyscale", "Image Operations", 0, 1, convert_to_grey)
cv2.createTrackbar("Sobel", "Image Operations", 0, 3, sobel)
cv2.createTrackbar("Laplace", "Image Operations", 0, 1, laplacian)
cv2.createTrackbar("Erode", "Image Operations", 0, 11, erode)
cv2.createTrackbar("Dilate", "Image Operations", 0, 11, dilate)
cv2.createTrackbar("Threshold", "Image Operations", 0, 255, threshold)

cv2.createTrackbar("Apply", "Image Operations", 0, 1, apply_operation)
cv2.createTrackbar("Save img", "Image Operations", 0, 1, save_img)
cv2.createTrackbar("Restart", "Image Operations", 0, 1, restart)


cv2.namedWindow('Preprocessing Operations')
cv2.resizeWindow("Preprocessing Operations", 300, 500) 
cv2.createTrackbar("Histogram eql", "Preprocessing Operations", 0, 1, equalise_histogram)
cv2.createTrackbar("Histogram stretching", "Preprocessing Operations", 0, 1, historgram_stretching)
cv2.createTrackbar("Subtract mean", "Preprocessing Operations", 0, 1, mean_subtraction)
cv2.createTrackbar("Color transfer", "Preprocessing Operations", 0, 1, color_transfer)
cv2.createTrackbar("Laplacian sharpening", "Preprocessing Operations", 0, 1, laplacian_sharpening)
cv2.createTrackbar("Sharpening", "Preprocessing Operations", 0, 1, sharpening)
cv2.createTrackbar("Mean noise removal", "Preprocessing Operations", 0, 1, noise_removal_mean)
cv2.createTrackbar("Median noise removal", "Preprocessing Operations", 0, 1, noise_removal_median)
cv2.createTrackbar("Color enhancement", "Preprocessing Operations", 0, 5, color_enhancement)
cv2.createTrackbar("Yellow", "Preprocessing Operations", 0, 1, find_yellow)
cv2.createTrackbar("Blue", "Preprocessing Operations", 0, 1, find_blue)

cv2.createTrackbar("Apply", "Preprocessing Operations", 0, 1, apply_operation)



trackbars = [[["HSV-color thresholds"], ["LH", "LS", "LV", "UH", "US", "UV", "Apply", "Save"]],
             [["Image Operations"], ["Greyscale", "Sobel", "Laplace", "Erode", "Dilate", "Threshold", "Apply", "Save img", "Restart"]],
             [['Preprocessing Operations'], ["Histogram eql", "Histogram stretching", "Subtract mean", "Color transfer", "Laplacian sharpening", "Sharpening", "Mean noise removal", "Median noise removal", "Color enhancement", "Yellow", "Blue", "Apply"]]]




#______________________________________#__________________________________________#


img_path = "Images\MultipleConesImages\yellow_cones_1.jpg" #'Images\FrameRemoved\Image_6.jpg'
output_path = "Images\\Other"
name_of_img = "test"


#______________________________________#__________________________________________#




img = cv2.imread(img_path)
processed_img = img.copy()
temp_img = img.copy()

while True:
    cv2.imshow('Original Image', img)
    cv2.imshow('Processed Image', processed_img)
    cv2.imshow('Temporary Image', temp_img)

    key = cv2.waitKey(1)
    if key == 27:
        break


cv2.destroyAllWindows()