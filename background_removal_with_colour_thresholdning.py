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

def find_yellow_cones_with_Hough(image):
    enhance_img = enhance_contrast(image)
    # 1. Load the image (assuming you already have the isolated yellow cones)
    img_yellow = colour_threshold(enhance_img, "HSV", [20,95,110], [35,255,255])
    min_area = 100
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2GRAY)

    # 3. Apply edge detection
    edges_1 = cv2.Canny(gray, 0, 250)  # You can adjust the thresholds as needed

    # 4. Post-processing (erosion and dilation)
    kernel = np.ones((3, 3), np.uint8)
    edges_2 = cv2.dilate(edges_1, kernel, iterations=1)
    edges_3 = cv2.erode(edges_2, kernel, iterations=1)

    # 5. Contour Detection
    contours, _ = cv2.findContours(edges_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Iterate through detected contours
    for contour in contours:
        # You can add additional filtering based on contour area, aspect ratio, etc.
        if cv2.contourArea(contour) > min_area:  # Adjust min_area as needed
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_yellow, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 7. Line Detection using Hough Line Transform
    lines = cv2.HoughLines(edges_3, 1, np.pi / 180, threshold=200)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_yellow, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw the detected lines

    cv2.imshow('Dark Lines', img_yellow)
    cv2.imshow('Edges', edges_3)

def find_dark_lines_in_yellow_cones(image):
    # 1. Load the image
    enhance_img = enhance_contrast(image)
    img_yellow = colour_threshold(enhance_img, "HSV", [20, 95, 110], [35, 255, 255])

    # 2. Convert to grayscale and ensure the correct data type (usually CV_8U)
    gray = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 0, 250)  # Adjust thresholds as needed

    # Convert the result to the appropriate data type
    #laplacian = np.uint8(np.absolute(laplacian))

    cv2.imshow('Yellow Cones', edges)
    #cv2.imshow('Dark Lines', edges)

find_dark_lines_in_yellow_cones(resized_image_2)
cv2.waitKey()
cv2.destroyAllWindows()