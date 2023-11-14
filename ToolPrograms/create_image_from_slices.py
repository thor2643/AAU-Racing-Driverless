import os
import random
import cv2
import numpy as np

# Define the path to the folder containing the image slices and the path to the background image
slices_folder = "Images\\YellowCones"
#slices_folder_2 = "Images\\BlueCones"
output_folder = "Images\\MultipleConesImages"
name = "yellow_cones_4"

# Get a list of all the image slices in the folder
slices = [os.path.join(slices_folder, f) for f in os.listdir(slices_folder) if os.path.isfile(os.path.join(slices_folder, f))] #+ [os.path.join(slices_folder_2, f) for f in os.listdir(slices_folder_2) if os.path.isfile(os.path.join(slices_folder_2, f))]

# Define the desired size of the final image
desired_size = (720, 480)

# Create an empty numpy array with the desired size of the final image and fill it with the background image
#background_image = cv2.imread(background_image_path)
final_image = np.zeros((desired_size[1], desired_size[0], 3), dtype=np.uint8)
#final_image[:, :] = background_image

# Keep track of the positions of the slices that have already been added to the final image
positions = []
min_positions = [[0,0]]
x_min = [0]

x = 0
y = 0

#cv2.imshow("image", final_image)

while True:

    #Try filling in a picture at the current position
    #If not possible -> update position
    success = False
    for i in range(20):
        # Pick a random image slice from the list
        slice_path = random.choice(slices)
        #slices.remove(slice_path)

        slice_image_1 = cv2.imread(slice_path)


        w_slice = slice_image_1.shape[1]
        h_slice = slice_image_1.shape[0]


        #Add padding if slice is too small
        if w_slice < 30:
            slice_image = cv2.copyMakeBorder(slice_image_1, 7,7,7,7, cv2.BORDER_REPLICATE)
            
            w_slice = slice_image.shape[1]
            h_slice = slice_image.shape[0]
        else:
            slice_image = slice_image_1
        


        vertical_space = True

        #Check for overlapping
        for position in positions:
            if any(position[1] <= i <= position[3] for i in range(y, y+h_slice)):
                if position[0] <= x+1 <= position[2]:
                    vertical_space = False
                    break

        #Check for image borders              
        if x+w_slice <= final_image.shape[1] and y+h_slice <= final_image.shape[0] and vertical_space == True:
            final_image[y:y+h_slice, x:x+w_slice] = slice_image

            positions.append([x, y, x+w_slice, y+h_slice])
            min_positions.append([x+w_slice, y])
            x_min.append(x+w_slice)

            success = True
            break
    
    
    idx = x_min.index(min(x_min))

    #If it was not possible to put slice in at current position, remove it and update the position
    #the position with the lowest x-value is chosen
    if not success:
        min_positions.pop(idx)
        x_min.pop(idx)

        if len(min_positions) == 0:
            break
        else:
            idx = x_min.index(min(x_min))

            x = min_positions[idx][0]+1
            y = min_positions[idx][1]

    else: 
        x = min_positions[idx][0]
        y += h_slice+1



path = output_folder + "\\" + name + ".jpg"

cv2.imwrite(path, final_image)    
cv2.imshow("image", final_image)
cv2.waitKey()
   

cv2.destroyAllWindows()


    

    
