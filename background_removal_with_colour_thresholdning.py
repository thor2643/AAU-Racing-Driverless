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


def colour_threshold(image, colourspace: str, lower_val: list, upper_val: list):
    if colourspace == "HSV":
        # convert to HSV
        colour_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    elif colourspace == "BGR":
        colour_image = image

    # set lower and upper colour limits
    temp_lower_val = np.array(lower_val)
    temp_upper_val = np.array(upper_val)

    # Threshold the HSV image to get only green colours
    mask = cv2.inRange(colour_image, temp_lower_val, temp_upper_val)

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

def enhance_contrast(image):
    # converting to LAB color space
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    result = np.hstack((image, enhanced_img))
    #cv2.imshow('Result', result)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    return result

def remove_image_frame(image):
    dist_to_horizontal_edge = 0
    dist_to_vertical_edge = 0
    height, width = image.shape[:2]

    x = int(width / 2)    
    for y in range(image.shape[0]):
        if not image[y, x].all():
            dist_to_horizontal_edge += 1
        else:
            break

    y = int(height / 2)
    for x in range(image.shape[1]):
        if not image[y, x].all():
            dist_to_vertical_edge += 1
        else:
            break
    
    sliced_image = image[dist_to_horizontal_edge:image.shape[0]-dist_to_horizontal_edge, dist_to_vertical_edge:image.shape[1]-dist_to_vertical_edge]
    
    return sliced_image



def laplacian_edge_detection(image):
    grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian filter to the grayscale image
    laplacian = cv2.Laplacian(grayscaled_image, cv2.CV_64F)

    return laplacian

    # Show the result
    #cv2.imshow('laplacian_image.png', laplacian)

def mean_image(image):
    # Define a kernel for local averaging
    kernel = np.ones((5, 5), np.float32) / 25  # A simple 5x5 averaging kernel

    # Apply convolution with the kernel
    averaged_image = cv2.filter2D(image, -1, kernel)
    brighter_image = image - averaged_image

    # Save or display the result
    #cv2.imshow('bright_image', brighter_image)
    return brighter_image

"""
img_blue = colour_threshold_HSV(resized_image_1, "img1", [80,95,110], [165,255,255])
img_yellow = colour_threshold_HSV(resized_image_1, "img2", [20,95,110], [35,255,255])

# Creating grayscale images
gray_img_blue = cv2.cvtColor(img_blue, cv2.COLOR_BGR2GRAY)
gray_img_yellow = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2GRAY)

# Set up a threshold mark for making binary images
_, binary_img_blue = cv2.threshold(gray_img_blue, 150, 255, cv2.THRESH_BINARY)
inv_binary_img_blue = 255-binary_img_blue

_, binary_img_yellow = cv2.threshold(gray_img_yellow, 240, 255, cv2.THRESH_BINARY)
inv_binary_img_yellow = 255-binary_img_yellow


cv2.imshow("Blue Cones", binary_img_blue)
cv2.imshow("Yellow Cones", binary_img_yellow)
cv2.waitKey()
cv2.destroyAllWindows()
"""
#img_yellow = colour_threshold_HSV(enhanced_image, "img2", [20,95,110], [35,255,255])
#cv2.imshow("Blue Cones", img_blue)
#cv2.imshow("White Cones", inv_enhanced_image)
##cv2.imshow("Sad Cones", inv_enhanced_image)
#cv2.imshow("White Cones", inv_enhanced_image_2)
##cv2.imshow("White Cones", enhanced_image_3)

def find_blue_cones(image):
    # Find all blue parts of the cone using HSV colour thresholding
    img_blue = colour_threshold(image, "HSV", [80,95,110], [165,255,255])
    
    # Find all the white lines on the cones by inverting the image to easily detect white colours with HSV colour thresholding
    inv_image = 255-image
    img_white_lines = colour_threshold(inv_image, "HSV", [0,0,0], [255,255,55])

    # Convert the images to greyscale to convert them to binary images
    gray_img_blue = cv2.cvtColor(img_blue, cv2.COLOR_BGR2GRAY)
    gray_img_white_lines = cv2.cvtColor(img_white_lines, cv2.COLOR_BGR2GRAY)

    # Convert the greyscaled images to binary images. 
    _, binary_img_blue = cv2.threshold(gray_img_blue, 210, 255, cv2.THRESH_BINARY)
    _, binary_img_white_lines = cv2.threshold(gray_img_white_lines, 140, 255, cv2.THRESH_BINARY)

    # The bitwise_and operator helps us combine the to images so that the white colours (now black blobs) that were previously missing from the blue cones (also black blobs) 
    # are combined with each other resulting in an image consisting of whole cones
    result = cv2.bitwise_and(binary_img_blue, binary_img_white_lines)
    result_image = 255-result

    # Create a kernel to apply opening (dilation) and closing (erosion) to the image, which will help connecting the black and white cone parts completely, 
    # as there are still a few pixels that need to be connected
    kernel = np.ones((3,3), np.uint8)
    opening_image = cv2.dilate(result_image, kernel, iterations= 1)
    closing_image = cv2.erode(opening_image, kernel, iterations= 1)

    #cv2.imshow("WIN?", closing_image)
    # Return the image
    return closing_image

def find_yellow_cones(image):
    enhance_img = enhance_contrast(image)

    gray_enhance_img = cv2.cvtColor(enhance_img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    # You can choose from different methods like cv2.ADAPTIVE_THRESH_MEAN_C or cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    # Block size defines the size of the neighborhood area for threshold calculation
    # C is a constant subtracted from the mean (or weighted mean in the case of Gaussian) value
    #result = cv2.adaptiveThreshold(gray_enhance_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=3, C=6)
    
    img_yellow = colour_threshold(enhance_img, "HSV", [20,95,110], [35,255,255])

    inv_image = 255-image
    img_black_lines = colour_threshold(enhance_img, "HSV", [10,75,0], [75,255,255]) #[0,0,0], [255,255,255], (enhance_img, "BGR", [50,55,120], [255,255,255])
    
    # Creating grayscale images
    gray_img_yellow = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2GRAY)
    gray_img_black = cv2.cvtColor(img_black_lines, cv2.COLOR_BGR2GRAY)
    
    result = cv2.adaptiveThreshold(gray_img_black, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=13, C=2)
    inv_result = 255-result

    # Set up a threshold mark for making binary images
    _, binary_img_yellow = cv2.threshold(gray_img_yellow, 245, 255, cv2.THRESH_BINARY)
    inv_binary_img_yellow = 255-binary_img_yellow
    eroded_result = cv2.dilate(result, kernel2, iterations= 1)
    background = np.zeros(eroded_result.shape, eroded_result.dtype)
    hej = cv2.bitwise_xor(eroded_result, inv_result, inv_result)

    #cv2.imshow("Something", hej)
    #cv2.imshow("Result", inv_result)
    #cv2.imshow("Yellow Cones", binary_img_yellow)
    cv2.imshow("Black Cones", img_black_lines)

def find_yellow_cones_with_laplacian(image):
    # 1. Load the image
    enhance_img = image #enhance_contrast(image)
    img_yellow = colour_threshold(enhance_img, "HSV", [20, 95, 110], [35, 255, 255])
    img_cone = colour_threshold(enhance_img, "BGR", [26, 32, 42], [100, 255, 255])
    img_orange_cones = colour_threshold(enhance_img, "BGR", [0, 0, 81], [163, 188, 255])
    # 2. Convert to grayscale
    gray_img_cone = cv2.cvtColor(img_cone, cv2.COLOR_BGR2GRAY)
    gray_img_orange_cones = cv2.cvtColor(img_orange_cones, cv2.COLOR_BGR2GRAY)

    _, binary_img_cone = cv2.threshold(gray_img_cone, 245, 255, cv2.THRESH_BINARY)
    _, binary_img_orange_cones = cv2.threshold(gray_img_orange_cones, 165, 255, cv2.THRESH_BINARY)

    # 3. Apply Laplacian to the img_yellow
    laplacian = cv2.Laplacian(img_yellow, cv2.CV_64F)
    laplacian_2 = cv2.convertScaleAbs(laplacian) 
    #print(laplacian_2.shape)
    # Resize Laplacian to match the dimensions of binary_img_cone
    gray_img_laplacian = cv2.cvtColor(img_cone, cv2.COLOR_BGR2GRAY)

    # Apply threshold to Laplacian result
    _, binary_img_laplacian = cv2.threshold(gray_img_laplacian, 100, 255, cv2.THRESH_BINARY)    

    # Perform bitwise OR operation with the binary cone mask
    summed_img = cv2.bitwise_and(binary_img_cone, binary_img_laplacian)
    kernel = np.ones((3,3), np.uint8)
    opening_image = cv2.erode(summed_img, kernel, iterations= 1)
    closing_image = cv2.dilate(opening_image, kernel, iterations= 1)
    inv_closing_img = 255 - closing_image
    removed_blobs_image = remove_blobs(inv_closing_img)
    final_img = remove_background_noise(removed_blobs_image)
    
    cv2.imshow('Orange Cones', img_orange_cones)
    cv2.imshow('Binary Orange Cones', binary_img_orange_cones)
    cv2.imshow('Background removal', final_img)

def remove_blobs(image):
    binary_mask = image

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

def remove_background_noise(image):
    # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image

    # Apply line detection (Hough Line Transform)
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, threshold=60, minLineLength=2, maxLineGap=30)

    if lines is not None:  # Check if lines were detected
        # Create a binary mask to remove the detected lines (stand)
        mask = np.ones_like(gray, dtype=np.uint8) * 255

        # Filter and remove the detected lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 0, thickness=2)  # Use 255 to draw lines

        # Apply the mask to remove the stand
        result = cv2.bitwise_and(image, image, mask=mask)

        return result
    else:
        return image  # No lines detected, return the original image


def remove_all_but_concrete(image, lower_yuv: list, upper_yuv: list):
    lower_yuv = np.array(lower_yuv, dtype=np.uint8)  # Convert to NumPy array
    upper_yuv = np.array(upper_yuv, dtype=np.uint8)  # Convert to NumPy array

    # 1. Convert the image to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # 2. Apply YUV thresholding to find concrete areas
    yuv_mask = cv2.inRange(yuv_image, lower_yuv, upper_yuv)

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
        concrete_area = image[y:y+h, x:x+w]

        return concrete_area

    # 7. If no concrete is found, return None
    return None

concrete = remove_all_but_concrete(resized_image_4, [103, 120, 125], [198, 133, 140])
#cv2.imshow("Concrete", concrete)

find_yellow_cones_with_laplacian(concrete)
cv2.waitKey()
cv2.destroyAllWindows()