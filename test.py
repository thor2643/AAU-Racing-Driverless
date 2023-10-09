import cv2
import numpy as np

"""

KERNELS TIL FORSKELLIGE TESTS

kernel1 = np.array([[0,0,1,0,0],
                    [0,0,1,0,0],
                    [1,1,1,1,1],
                    [0,0,1,0,0],
                    [0,0,1,0,0]], np.uint8)

kernel2 = np.array([[1,0,1],
                    [0,0,0],
                    [1,0,1]], np.uint8)

kernel3 = np.ones((3,3), np.uint8)

kernel4 = np.array([[1,1],
                    [1,1],
                    [1,1],
                    [1,1],
                    [1,1],
                    [1,1]], np.uint8)

kernel5 = np.array([[1,1],
                    [1,1],
                    [1,1],
                    [1,1],
                    [1,1],
                    [0,0],
                    [0,0],
                    [0,0],
                    [0,0],
                    [0,0],
                    [1,1],
                    [1,1],
                    [1,1],
                    [1,1],
                    [1,1],
                    [1,1]], np.uint8)

"""


img_1 = cv2.imread('Race track with cones.png')

def colour_threshold_HSV(image, name: str, lower_val: list, upper_val: list):
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

    # set lower and upper colour limits
    temp_lower_val = np.array(lower_val)
    temp_upper_val = np.array(upper_val)

    # Threshold the HSV image to get only green colours
    mask = cv2.inRange(hsv, temp_lower_val, temp_upper_val)

    # apply mask to original image - this shows the green with black blackground
    only_green = cv2.bitwise_and(image, image, mask = mask)

    # create a black image with the dimensions of the input image
    background = np.zeros(image.shape, image.dtype)
    # invert to create a white image
    background = cv2.bitwise_not(background)
    # invert the mask that blocks everything except green -
    # so now it only blocks the green area's
    mask_inv = cv2.bitwise_not(mask)
    # apply the inverted mask to the white image,
    # so it now has black where the original image had green
    masked_bg = cv2.bitwise_and(background, background, mask = mask_inv)
    # add the 2 images together. It adds all the pixel values, 
    # so the result is white background and the the green from the first image
    final = cv2.add(only_green, masked_bg)
    
    #show image
    #cv2.imshow(name, final)
    return final

img = colour_threshold_HSV(img_1, "img2", [20,95,110], [80,255,255])

# Read the image and apply thresholding
image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(image_gray, 100, 200)

# Define a kernel for filtering
kernel = np.array([[-1,-2,-1],
                   [0,0,0],
                   [1,2,1]], np.uint8)

# Apply filter to blend the cones together
blended = cv2.filter2D(edges, -1, kernel)

# Display the result
cv2.imshow('Blended Image', blended)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

TING DER ER PRØVER AF (UNDSKYLD AT DET ER MEGET RODET!)

### SOBEL
# Convert the image to grayscale
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
_, binary_gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

gray_cropped = gray[200:230,620:645] 
# Apply Sobel operator to the grayscale image
sobelx = cv2.Sobel(binary_gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(binary_gray, cv2.CV_64F, 0, 1)

# Combine the results of Sobel operator
sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

#erosion_sobelx = cv2.erode(sobelx, kernel3, iterations=1)
dilation_sobelx = cv2.dilate(sobelx, kernel3, iterations=1)

## LAPLACIAN!
# Apply Laplacian filter to the grayscale image
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Save the result
cv2.imwrite('laplacian_image.png', laplacian)

# Save the result
cv2.imwrite('sobel_image.png', dilation_sobelx)
### SOBEL

### CANNY
# Apply edge detection to create a binary image
edges = cv2.Canny(gray, 100, 200)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Fill the contours with white color
cv2.drawContours(gray, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Save the result
cv2.imwrite('filled_cones.png', gray)
### CANNY

###
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Set up a threshold mark
_, binary_img1 = cv2.threshold(gray_img1, 150, 255, cv2.THRESH_BINARY)
inv_binary_img1 = 255-binary_img1

_, binary_img2 = cv2.threshold(gray_img2, 150, 255, cv2.THRESH_BINARY)
inv_binary_img2 = 255-binary_img2

#PATH for img2
dilation_1 = cv2.dilate(sobelx, kernel4, iterations=2)
#erosion_1 = cv2.erode(dilation_1, kernel3, iterations=1)
dilation_2 = cv2.dilate(dilation_1, kernel5, iterations=1)
erosion_1 = cv2.erode(dilation_1, kernel2, iterations=1)
#erosion_2 = cv2.erode(dilation_2, kernel3, iterations=1)

###

#cv2.imshow("Original image",img1)
#cv2.imshow("Dilated image",img_dilation)

#cv2.imwrite("Image 1.png", inv_binary_img1)
#cv2.imwrite("Image 2.png", inv_binary_img2)


cv2.imshow("Original image", gray_img2)
cv2.imshow("Dilated image", erosion_1)
#cv2.imshow("Binary image", erosion_img1_v2)


cv2.waitKey(0)
cv2.destroyAllWindows()
"""





"""
def assemble_image(image1, image2):
    # Resize the images if necessary
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    # Create a four-channel image with transparency
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2BGRA)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2BGRA)

    # Blend the images using addWeighted() method
    alpha = 0.5
    beta = 1 - alpha
    result_img = cv2.addWeighted(img1, alpha, img2, beta, 0)

    # Save the blended image

    img1_normalized = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
    img2_normalized = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)

    alpha = 0.5  # Adjust the alpha value to control the blending ratio
    beta = 0.5
    result_img = cv2.addWeighted(img1_normalized, alpha, img2_normalized, beta, 0)

    #result_img = image1 + image2

    #image1[:,:,0]=(image2[:,:,0]<125)*np.uint8(image2[:,:,0])
    #image1[:,:,1]=(image2[:,:,1]<125)*np.uint8(image2[:,:,1])
    #image1[:,:,2]=(image2[:,:,2]<255)*np.uint8(image2[:,:,2])

    #result_img = image1

    cv2.imshow('Result', result_img) 
    return result_img

"""
