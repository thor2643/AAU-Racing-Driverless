import numpy as np
import cv2
import os


def click_event(event, x, y, flags, param):
    global coords
    global shape

    #printer koordinaterne ved venstreklik
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(len(coords)) #str(x) laver det om til en string 

        coords.append([[int(x-shape[0]/2), int(y-shape[1]/2)],
                       [int(x+shape[0]/2), int(y+shape[1]/2)]])

        cv2.rectangle(img_shown, (coords[len(coords)-1][0][0], coords[len(coords)-1][0][1]), (coords[len(coords)-1][1][0], coords[len(coords)-1][1][1]), (0, 0, 255), 2)


        cv2.imshow('image', img_shown)



#        w    h
shape = [32, 64]  
input_folder = "Images\\Raw"
output_folder = "Images\\Other"

general_name = "Temporary_cone_"

#128x64

j = 0
for i, filename in enumerate(os.listdir(input_folder)):
    coords = []

    #gets the RGB image from input folder
    img = cv2.imread(os.path.join(input_folder, filename))
    img_shown = img.copy()

    cv2.imshow('image', img_shown) #vinduenavnet skal være ens

    cv2.setMouseCallback('image', click_event) #kalder vores funktion ved museaktivitet

    key = cv2.waitKey(0)

    print(coords)
    for coord in coords:
        img_slice = img[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]]

        path = output_folder + "\\" + general_name + str(j) + ".jpg" #os.path.join(output_folder, general_name, str(i), ".jpg")

        cv2.imwrite(path, img_slice)

        j+=1

    cv2.destroyAllWindows()

    if key == ord("q"):
        break

