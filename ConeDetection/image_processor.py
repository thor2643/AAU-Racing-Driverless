import numpy as np
import cv2


class ImageProcessor():
    def __init__(self) -> None:
        pass
        
    def remove_image_frame(self, image):
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

    def remove_all_but_concrete(self, img, lower_yuv: list, upper_yuv: list):
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

    def remove_blobs(self, img):
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

