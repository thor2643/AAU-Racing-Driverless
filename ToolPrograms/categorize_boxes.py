import cv2
import os

# Path to folder containing images
input_folder_path = "Hog//Cones//ekstra"
output_folder_path_yellow = "Hog//Cones//Yellow"
output_folder_path_blue = "Hog//Cones//Blue"
output_folder_path_negative = "Hog//Cones//NegativeSamples"


print("y = yellow, b = blue, n = negative, q = quit")
# runs through all images in the folder
for filename in os.listdir(input_folder_path):
    # if the file is an image
    if filename.endswith(".jpg" or ".png"):
        # read the image
        image = cv2.imread(os.path.join(input_folder_path, filename))
        # show the image
        cv2.imshow("Image", image)
        # waiting for caracterisation
        key = cv2.waitKey(0)
        # if the key pressed is "y"
        if key == ord("y"):
            #chance name of image to yellow
            filename = filename.replace(".jpg", "_yellow.jpg")
            # save the image in the yellow folder
            cv2.imwrite(os.path.join(output_folder_path_yellow, filename), image)
            
        # if the key pressed is "b"
        elif key == ord("b"):
            #chance name of image to blue
            filename = filename.replace(".jpg", "_blue.jpg")
            # save the image in the blue folder
            cv2.imwrite(os.path.join(output_folder_path_blue, filename), image)
            
        # if the key pressed is "n"
        elif key == ord("n"):
            #chance name of image to negative
            filename = filename.replace(".jpg", "_negative1.jpg")
            # save the image in the negative folder
            cv2.imwrite(os.path.join(output_folder_path_negative, filename), image)
            
        # if the key pressed is "q"
        elif key == ord("q"):
            # break the loop
            break
        # if any other key is pressed
        else:
            print("Invalid key pressed. y = yellow, b = blue, n = negative, q = quit")
