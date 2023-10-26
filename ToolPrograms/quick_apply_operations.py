import numpy as np
import cv2

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

def equalise_histogram(x):
    global processed_img, temp_img

    # convert it to grayscale
    img_yuv = cv2.cvtColor(processed_img,cv2.COLOR_BGR2YUV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

    #convert back
    temp_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def mean_subtraction(x):
    global processed_img, temp_img

    # convert it to grayscale
    img_yuv = cv2.cvtColor(processed_img,cv2.COLOR_BGR2HSV)

    #Gaussian filter
    mean_filter = 1/16 * np.array([[1,2,1],
                                    [2,4,2],
                                    [1,2,1]])

        
    img_avg = cv2.filter2D(img_yuv[:,:,0],-1, mean_filter)

    img_yuv[:,:,0] =cv2.subtract(img_yuv[:,:,0], img_avg)

    #convert back
    temp_img = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)

    


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
cv2.resizeWindow("Image Operations", 300, 375) 
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
cv2.resizeWindow("Image Operations", 300, 200) 
cv2.createTrackbar("Histogram eql", "Preprocessing Operations", 0, 1, equalise_histogram)
cv2.createTrackbar("Subtract mean", "Preprocessing Operations", 0, 1, mean_subtraction)

cv2.createTrackbar("Apply", "Preprocessing Operations", 0, 1, apply_operation)



trackbars = [[["HSV-color thresholds"], ["LH", "LS", "LV", "UH", "US", "UV", "Apply", "Save"]],
             [["Image Operations"], ["Greyscale", "Sobel", "Laplace", "Erode", "Dilate", "Threshold", "Apply", "Save img", "Restart"]],
             [['Preprocessing Operations'], ["Histogram eql", "Subtract mean"]]]




#______________________________________#__________________________________________#


img_path = 'Images\FrameRemoved\Image_6.jpg'
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