import numpy as np 
import cv2

# load image
img_1 = cv2.imread("cone_tracking_data\\amz_00052.jpg")
img_2 = cv2.imread("cone_tracking_data\\amz_00109.jpg")
img_3 = cv2.imread("cone_tracking_data\\amz_00161.jpg")
img_4 = cv2.imread("cone_tracking_data\\amz_00218.png")
img_5 = cv2.imread("cone_tracking_data\\amz_00296.png")
img_6 = cv2.imread("cone_tracking_data\\amz_00508.jpg")
img_7 = cv2.imread("cone_tracking_data\\amz_00633.jpg")
img_8 = cv2.imread("cone_tracking_data\\amz_00702.png")
img_9 = cv2.imread("cone_tracking_data\\amz_00770.png")
img_10 = cv2.imread("cone_tracking_data\\amz_00799.jpg")

resolution_x = 620
resolution_y = 480

resized_image_1 = cv2.resize(img_1, (resolution_x, resolution_y))
resized_image_2 = cv2.resize(img_2, (resolution_x, resolution_y))
resized_image_3 = cv2.resize(img_3, (resolution_x, resolution_y))
resized_image_4 = cv2.resize(img_4, (resolution_x, resolution_y))
resized_image_5 = cv2.resize(img_5, (resolution_x, resolution_y))
resized_image_6 = cv2.resize(img_6, (resolution_x, resolution_y))
resized_image_7 = cv2.resize(img_7, (resolution_x, resolution_y))
resized_image_8 = cv2.resize(img_8, (resolution_x, resolution_y))
resized_image_9 = cv2.resize(img_9, (resolution_x, resolution_y))
resized_image_10 = cv2.resize(img_10, (resolution_x, resolution_y))

print(img_1.shape)
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

"""
img_blue = colour_threshold_HSV(img_1, "img1", [80,95,110], [165,255,255])
img_yellow = colour_threshold_HSV(img_1, "img2", [20,95,110], [80,255,255])

# Creating grayscale images
gray_img1 = cv2.cvtColor(img_blue, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2GRAY)

# Set up a threshold mark for making binary images
_, binary_img1 = cv2.threshold(gray_img1, 150, 255, cv2.THRESH_BINARY)
inv_binary_img1 = 255-binary_img1

_, binary_img2 = cv2.threshold(gray_img2, 150, 255, cv2.THRESH_BINARY)
inv_binary_img2 = 255-binary_img2
"""
cv2.imwrite("Thor_data/Image_1.jpg", resized_image_1)
cv2.imwrite("Thor_data/Image_2.jpg", resized_image_2)
cv2.imwrite("Thor_data/Image_3.jpg", resized_image_3)
cv2.imwrite("Thor_data/Image_4.png", resized_image_4)
cv2.imwrite("Thor_data/Image_5.png", resized_image_5)
cv2.imwrite("Thor_data/Image_6.jpg", resized_image_6)
cv2.imwrite("Thor_data/Image_7.jpg", resized_image_7)
cv2.imwrite("Thor_data/Image_8.png", resized_image_8)
cv2.imwrite("Thor_data/Image_9.png", resized_image_9)
cv2.imwrite("Thor_data/Image_10.jpg", resized_image_10)
cv2.waitKey()
cv2.destroyAllWindows()