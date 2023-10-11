import cv2
from ConeDetector import ConeDetector


Detector = ConeDetector()

img = cv2.imread("Thor_data\\Image_2.jpg", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("Thor_data\\Image_2.jpg")

#img_blue = colour_threshold_HSV(img_1, "img1", [80,95,110], [165,255,255])
img_yellow, img_binary_yellow = Detector.colour_threshold_HSV(img1, "img2", [20,95,110], [35,255,255])

blobs, blob_features = Detector.get_blobs_and_features(img_binary_yellow)

thresholds = {"min_area": 15,
              "min_perimeter": 2,
              "min_aspect_ratio": 0.8,
              "min_extent": 0.3,
              "max_extent": 0.8,
              "solidity": 0.4,
              "mass_center_height_ratio": 0.6}


cones_unsorted = Detector.sort_blobs(thresholds, blobs, blob_features)

cones = Detector.remove_cone_top_blobs(cones_unsorted)

img1 = Detector.draw_bbox_CoM(img1, cones)

cv2.imshow("Cone Detector Pure", img_yellow)
cv2.imshow("Cone Detector", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()




