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

test_img_blue_cones = cv2.imread("Images\\MultipleConesImages\\blue_cones_1.jpg")
test_img_yellow_cones = cv2.imread("Images\\MultipleConesImages\\yellow_cones_1.jpg")

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



kernel1 = np.array([[0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0],
                    [0,1,1,1,1,1,0],
                    [0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0]], np.uint8)

kernel2 = np.array([[0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,1,1,0,0,0],
                    [0,0,1,1,0,0,0],
                    [0,0,1,1,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0]], np.uint8)

kernel3 = np.array([[0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,1,1,1,1,1,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0]], np.uint8)


def colour_threshold(img_BGR, lower_val: list, upper_val: list, colourspace="HSV"):
    if colourspace == "HSV":
        colour_img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV) 
    elif colourspace == "BGR":
        colour_img = img_BGR
    elif colourspace == "YUV":
        colour_img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV)

    # set lower and upper colour limits
    temp_lower_val = np.array(lower_val)
    temp_upper_val = np.array(upper_val)

    # Threshold the image to get a desired colour range
    mask = cv2.inRange(colour_img, temp_lower_val, temp_upper_val)

    # apply mask to original image. This shows the desired colour range with a black blackground
    desired_colours = cv2.bitwise_and(img_BGR, img_BGR, mask = mask)

    # create a white image with the dimensions of the input image. Scale with 255 to set all pixel values to 255 instead of 1, because that would result in a black image.
    background = np.ones(img_BGR.shape, img_BGR.dtype)*255

    # invert the mask that blocks everything except the desired colour range
    mask_inv = cv2.bitwise_not(mask)
    # apply the inverted mask to the white image
    masked_background = cv2.bitwise_and(background, background, mask = mask_inv)
    # add the 2 images together. This yields an image with white background and the desired colours shown
    final = cv2.add(desired_colours, masked_background)
    
    #show image
    #cv2.imshow(name, final)
    return final

def find_blue_cones(img):
    # Find all blue parts of the cone using HSV colour thresholding
    img_blue = colour_threshold(img, [80,95,110], [165,255,255])
    
    # Find all the white lines on the cones by inverting the image to easily detect white colours with HSV colour thresholding
    inv_img = 255-img
    img_white_lines = colour_threshold(inv_img, [0,0,0], [255,255,55])

    # Convert the images to greyscale to convert them to binary images
    gray_img_blue = cv2.cvtColor(img_blue, cv2.COLOR_BGR2GRAY)
    gray_img_white_lines = cv2.cvtColor(img_white_lines, cv2.COLOR_BGR2GRAY)

    # Convert the greyscaled images to binary images. 
    _, binary_img_blue = cv2.threshold(gray_img_blue, 210, 255, cv2.THRESH_BINARY)
    _, binary_img_white_lines = cv2.threshold(gray_img_white_lines, 140, 255, cv2.THRESH_BINARY)

    # The bitwise_and operator helps us combine the to images so that the white colours (now black blobs) that were previously missing from the blue cones (also black blobs) 
    # are combined with each other resulting in an image consisting of whole cones
    result = cv2.bitwise_and(binary_img_blue, binary_img_white_lines)
    result_img = 255-result

    # Create a kernel to apply opening (dilation) and closing (erosion) to the image, which will help connecting the black and white cone parts completely, 
    # as there are still a few pixels that need to be connected
    kernel = np.ones((3,3), np.uint8)
    opening_img = cv2.dilate(result_img, kernel, iterations= 1)
    closing_img = cv2.erode(opening_img, kernel, iterations= 1)
    
    cv2.imshow("Blue Cones", closing_img)
    # Return the image
    return closing_img

def find_orange_cones(img):
    img_orange_cones = colour_threshold(img, [0, 0, 81], [163, 188, 255], "BGR")

    gray_img_orange_cones = cv2.cvtColor(img_orange_cones, cv2.COLOR_BGR2GRAY)

    _, binary_img_orange_cones = cv2.threshold(gray_img_orange_cones, 165, 255, cv2.THRESH_BINARY)  

    cv2.imshow('Binary Orange Cones', binary_img_orange_cones)

def find_yellow_cones(img):
    # This was the original threshold method, where we tried to find all yellow and black colours separately
    #img_yellow = colour_threshold(enhance_img, [20, 95, 110], [35, 255, 255]) 

    # Threshold the image to look for the complete cone as much as possible
    img_cone = colour_threshold(img, [26, 32, 42], [100, 255, 255], "BGR")
    
    # Convert to grayscale and then to binary
    gray_img_cone = cv2.cvtColor(img_cone, cv2.COLOR_BGR2GRAY)
    _, binary_img_cone = cv2.threshold(gray_img_cone, 245, 255, cv2.THRESH_BINARY)

    # Use opening and closing on the image to complete the cones
    kernel = np.ones((3,3), np.uint8)
    opening_img = cv2.erode(binary_img_cone, kernel, iterations= 1)
    closing_img = cv2.dilate(opening_img, kernel, iterations= 1)

    # Invert the image to get black cones on a white background and remove any larger blobs that are not cones
    inv_closing_img = 255 - closing_img
    final_img = remove_blobs(inv_closing_img)

    # Display the result
    cv2.imshow('Background removal', final_img)
    return final_img

def find_yellow_cones_with_laplacian(img):
    # Threshold images
    img_yellow = colour_threshold(img, [20, 95, 110], [35, 255, 255])
    img_cone = colour_threshold(img, [26, 32, 42], [100, 255, 255], "BGR")
   
    # Convert to grayscale and then to binary
    gray_img_cone = cv2.cvtColor(img_cone, cv2.COLOR_BGR2GRAY)
    _, binary_img_cone = cv2.threshold(gray_img_cone, 245, 255, cv2.THRESH_BINARY)
  
    BGR_img_yellow = cv2.cvtColor(img_yellow, cv2.COLOR_HSV2BGR) 
    
    # Apply Laplacian to img_yellow
    laplacian = cv2.Laplacian(BGR_img_yellow, cv2.CV_64F)
    laplacian_2 = cv2.convertScaleAbs(laplacian) 
    #print(laplacian_2.shape)
    # Resize Laplacian to match the dimensions of binary_img_cone
    gray_img_laplacian = cv2.cvtColor(laplacian_2, cv2.COLOR_BGR2GRAY)

    # Apply threshold to Laplacian result
    _, binary_img_laplacian = cv2.threshold(gray_img_laplacian, 100, 255, cv2.THRESH_BINARY)
    inv_binary_img_laplacian = 255-binary_img_laplacian
    
    # Perform bitwise_and operation with the binary cone mask
    summed_img = cv2.bitwise_and(binary_img_cone, inv_binary_img_laplacian)

    # Use opening and closing on the image to complete the cones
    kernel = np.ones((3,3), np.uint8)
    opening_img = cv2.erode(summed_img, kernel, iterations= 1)
    closing_img = cv2.dilate(opening_img, kernel, iterations= 1)

    # Invert the image to get black cones on a white background and remove any larger blobs that are not cones
    inv_closing_img = 255 - closing_img
    final_img = remove_blobs(inv_closing_img)
    
    # Display the result
    cv2.imshow('Yellow Cones', final_img)
    return final_img

def remove_blobs(img):
    binary_mask = img

    # Find blobs and filter based on area
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_blob_area = 500  

    clean_mask = np.zeros_like(binary_mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_blob_area:
            cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return clean_mask
    #cv2.imshow("Large background blobs removed", clean_mask)

def remove_all_but_concrete(img, lower_yuv: list, upper_yuv: list):
    lower_yuv = np.array(lower_yuv, dtype=np.uint8)  # Convert to NumPy array
    upper_yuv = np.array(upper_yuv, dtype=np.uint8)  # Convert to NumPy array

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
        x, y, w, h = cv2.boundingRect(largest_blob)

        # 6. Crop the original image using the bounding box
        concrete_area = img[y:y+h, x:x+w]

        return concrete_area

    # 7. If no concrete is found, return None
    return None

#concrete = remove_all_but_concrete(resized_image_2, [103, 120, 125], [198, 133, 140])
#cv2.imshow("Concrete", concrete)

find_blue_cones(test_img_blue_cones)

cv2.waitKey()
cv2.destroyAllWindows()