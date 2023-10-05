import numpy as np 
import cv2

# load image
img_1 = cv2.imread("Race track with cones.png")
img_2 = cv2.imread("Race track with cones 2.png")
#img_3 = cv2.imread("Race track with cones 2.png")

kernel1 = np.array([[0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0]], np.uint8)
kernel2 = np.array([[0,0,0],
                   [0,0,0],
                   [0,0,0]], np.uint8)

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

    """
    img1_normalized = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
    img2_normalized = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)

    alpha = 0.5  # Adjust the alpha value to control the blending ratio
    beta = 0.5
    result_img = cv2.addWeighted(img1_normalized, alpha, img2_normalized, beta, 0)
    """
    #result_img = image1 + image2

    #image1[:,:,0]=(image2[:,:,0]<125)*np.uint8(image2[:,:,0])
    #image1[:,:,1]=(image2[:,:,1]<125)*np.uint8(image2[:,:,1])
    #image1[:,:,2]=(image2[:,:,2]<255)*np.uint8(image2[:,:,2])

    #result_img = image1

    cv2.imshow('Result', result_img) 

img1 = colour_threshold_HSV(img_1, "img1", [80,95,110], [165,255,255])
img2 = colour_threshold_HSV(img_1, "img2", [20,95,110], [80,255,255])

###
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Set up a threshold mark
_, binary_img1 = cv2.threshold(gray_img1, 150, 255, cv2.THRESH_BINARY)
inv_binary_img1 = 255-binary_img1

_, binary_img2 = cv2.threshold(gray_img2, 150, 255, cv2.THRESH_BINARY)
inv_binary_img2 = 255-binary_img2

#img_dilation = cv2.dilate(binary_img1, kernel1, iterations=1)
#img_erosion = cv2.erode(img_dilation, kernel, iterations=1)

###

#cv2.imshow("Original image",img1)
#cv2.imshow("Dilated image",img_dilation)
cv2.imwrite("Image 1.png", inv_binary_img1)
cv2.imwrite("Image 2.png", inv_binary_img2)
#assemble_image(img1, img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
