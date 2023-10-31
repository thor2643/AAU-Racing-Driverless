import numpy as np
import cv2

def nothing(x): #dummy fuction
    pass

# Add a "Save" button to the window
def save_values(x):
    if x == 1:
        values = [cv2.getTrackbarPos("LH", 'tracking'),
                  cv2.getTrackbarPos("LS", 'tracking'),
                  cv2.getTrackbarPos("LV", 'tracking'),
                  cv2.getTrackbarPos("UH", 'tracking'),
                  cv2.getTrackbarPos("US", 'tracking'),
                  cv2.getTrackbarPos("UV", 'tracking')]
        print("Values saved:", values)

#cap = cv2.VideoCapture(0)

cv2.namedWindow('tracking')
cv2.createTrackbar("LH", "tracking", 0, 255, nothing) #LH = lower Hue
cv2.createTrackbar("LS", "tracking", 0, 255, nothing) #LS = lower saturation
cv2.createTrackbar("LV", "tracking", 0, 255, nothing) #Value
cv2.createTrackbar("UH", "tracking", 255, 255, nothing) #U =  upper
cv2.createTrackbar("US", "tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "tracking", 255, 255, nothing)
cv2.createTrackbar("Save", "tracking", 0, 1, save_values)
    

while True:
    frame = cv2.imread('Images\\MultipleConesImages\\blue_cones_2.jpg') #hvis du vil se konceptet med et billede

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", 'tracking') #får fat i værdien af vores trackbars
    l_s = cv2.getTrackbarPos("LS", 'tracking')
    l_v = cv2.getTrackbarPos("LV", 'tracking')

    u_h = cv2.getTrackbarPos("UH", 'tracking')
    u_s = cv2.getTrackbarPos("US", 'tracking')
    u_v = cv2.getTrackbarPos("UV", 'tracking')

    l_b = np.array([l_h, l_s, l_v]) #nederst grænse for blå farve
    u_b = np.array([u_h, u_s, u_v]) #øverste grænse

    mask= cv2.inRange(hsv, l_b, u_b) #finder ud om hsv billede har dele som ligger i intervallet angivet af l_b og u_b

    res = cv2.bitwise_and(frame, frame, mask=mask)




    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)


    #cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


cv2.destroyAllWindows()