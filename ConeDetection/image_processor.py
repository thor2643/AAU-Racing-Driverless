import numpy as np
import cv2


class ImageProcessor():
    def __init__(self) -> None:
        pass
        


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

