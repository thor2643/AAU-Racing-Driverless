import os
import cv2
import numpy as np
import pickle

from ConeDetector import ConeDetector

Detector = ConeDetector()

# if you want it to process the image for blue cones, set Blue to True, else set it to False
def process_image(input_path, Blue, name):
    # Read image
    img = cv2.imread(f"{input_path}", cv2.IMREAD_COLOR)

    # Apply function to image
    if Blue == True:
        _, processed_img = Detector.colour_threshold_HSV(img, name + "_blue" , [80,95,110], [165,255,255])
    else:
        _, processed_img = Detector.colour_threshold_HSV(img, name + "_yellow", [20,95,110], [80,255,255])

    return processed_img

#creates a folder for the processed images, that originates from the input folder,
#and saves them in the new folder.
def process_folder(input_folder, output_folder):
    # Loop through all files in input folder
    for filename in os.listdir(input_folder):
        # Check if file is an image
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Get input and output paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder)

            # Create directory if it does not exist
            if not os.path.exists(input_path):
                os.makedirs(input_path)
                
            if not os.path.exists(output_path):
                os.makedirs(output_path) 

            # Process image and save to output folder
            processed_imgBlue=process_image(input_path, True, filename)
            processed_imgYellow=process_image(input_path, False, filename)
            
            cv2.imwrite(f'ConeDetection\\BinaryImages_img\\blue_{filename}', processed_imgBlue)
            cv2.imwrite(f'ConeDetection\\BinaryImages_img\\yellow_{filename}', processed_imgYellow)


            
#runs throgh the pictures in the desired folder, and gets BLOB features for each picture and stores them in array.
def get_BLOBS_and_cone_array(folder,filename):
    n=0
    
    #der var: for filename in os.listdir(folder):
    img=cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)
    contours, cont_props = Detector.get_blobs_and_features(img, method = cv2.CHAIN_APPROX_SIMPLE)
         
    return img, contours, cont_props

def sort_blobs_area(cont_props, min_area):
    sorted_cont_props=[]
    for cont in cont_props:
        w = cont[6]
        h = cont[7] 
        area = w*h
        if area > min_area:
            sorted_cont_props.append(cont)
    return sorted_cont_props

#shows image and asks if it is a cone or not and returns True or False
def cone_or_not(img,img_BGR):
    # Resize the image to make it larger
    img_BGR = cv2.resize(img_BGR, (0,0), fx=4, fy=4)
    cv2.imshow("img", img)
    cv2.imshow("img_RGB", img_BGR)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == ord('y'):
        return 1 #True
    elif key == ord('n'):
        return 0  #False
    else:
        print("Invalid input please try again. y=cone, n=not cone")
        return cone_or_not(img, img_RGB)
        
     
input_path = "Thor_data"
output_path = "ConeDetection\BinaryImages_img"
data=np.array([]) 
process_folder(input_path, output_path)
folder=output_path
#makes a list of the filenames in the input folder so we can iterate through them
filenames_in_input_folder = os.listdir(input_path)
file_num=0
#goes through each image in folder and gets features for each blob in the image,
# and asks if it is a cone or not.
for filename in os.listdir(folder):
    img, contours, cont_props= get_BLOBS_and_cone_array(folder, filename)
    i=0  
    sorted_cont_props=sort_blobs_area(cont_props,min_area=50)
    #crops out the images so we can classify them
    for cont in sorted_cont_props:
        w = cont[6]
        h = cont[7] 
        x1 = cont[9]
        y1 = cont[10]

        cropped_img_BLOB = img[y1:y1+h, x1:x1+w]
        
        #gets the RGB image from input folder
        img_RGB=cv2.imread(os.path.join(input_path, filenames_in_input_folder[file_num]), cv2.IMREAD_UNCHANGED)
        cropped_img_RGB = img_RGB[y1:y1+h, x1:x1+w]
            
        is_cone = cone_or_not(cropped_img_BLOB,cropped_img_RGB)
        sorted_cont_props[i] = [[sorted_cont_props[i]], [is_cone]]
        i+=1
    
    #saves the data in data array for each image
    data = [[data], [sorted_cont_props]]
    
    #we now look at the next image in the folder therefore we add 1 to file_num
    file_num+=1
    #to take account for the difference in number of images in the input folder and the output folder
    #10 RGB vs 20 binarynnnnnnn
    if file_num == len(os.listdir(input_path)):
        file_num=0
    print("new img")
        
# Save the array to a file
with open('my_array.pkl', 'wb') as file:
    pickle.dump(data, file)

with open('my_array.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
    print(loaded_data)