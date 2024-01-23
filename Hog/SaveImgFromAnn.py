import os
import yaml
import cv2
import numpy as np
from tqdm import tqdm

#Class id dictionary
class_id_dict = {
    0: "Yellow",
    1: "Blue",
    2: "Orange",
    3: "LargeOrange",
}           

count = 0

Extra = False

ResizeToClosestWindow = False

window_sizes = [(32,16), (64,32), (128,64)]

min_area = 1

def random_shift(image, x1, y1, x2, y2, shift = False):
    if shift:
        #Calculate max shift
        max_shift_x = int(abs(x2 - x1) * 0.25)
        max_shift_y = int(abs(y2 - y1) * 0.25)

        # Calculate random shifts in both x and y directions
        shift_x = np.random.randint(-max_shift_x, max_shift_x + 1)
        shift_y = np.random.randint(-max_shift_y, max_shift_y + 1)

         # Update the bounding box coordinates
        new_x1 = max(0, x1 + shift_x)
        new_y1 = max(0, y1 + shift_y)
        new_x2 = min(image.shape[1], x2 + shift_x)
        new_y2 = min(image.shape[0], y2 + shift_y)
         # Calculate the necessary adjustments to ensure the entire bounding box is included
        adjustment_x = max(0, x1 - new_x1, new_x2 - x2)
        adjustment_y = max(0, y1 - new_y1, new_y2 - y2)

        # Apply the adjustments to the new coordinates
        new_x1 += adjustment_x
        new_y1 += adjustment_y
        new_x2 += adjustment_x
        new_y2 += adjustment_y

        # Check if the new bounding box coordinates are valid
        if new_x1 >= new_x2 or new_y1 >= new_y2:
            return None
        # Crop the image using the updated bounding box coordinates
        cropped_image = image[new_y1:new_y2, new_x1:new_x2]

        return cropped_image
    else:
        return image[y1:y2, x1:x2]

def get_coordinates_from_txt(filename, img_shape):
    with open(filename, 'r') as f:
        lines = f.readlines()
        coordinates = []
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.split())
            # Convert normalized coordinates to pixel coordinates
            x_center *= img_shape[1]
            y_center *= img_shape[0]
            width *= img_shape[1]
            height *= img_shape[0]
            # Convert center coordinates to corner coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            #Calculate the hight and width
            h_w = (y2 - y1,x2 - x1) 

            if ResizeToClosestWindow:
                # Check which window size it is closest too in the window_sizes list, in regards to distance
                closest_window_size = min(window_sizes, key=lambda x: abs(x[0] - h_w[0]) + abs(x[1] - h_w[1]))
                difference = (closest_window_size[0] - h_w[0], closest_window_size[1] - h_w[1])

                # Calculate the new coordinates. Add half the difference to both x and y coordinates. 
                # This will center the window around the cone if x1 is smaller than x2 and y1 is smaller than y2¨
                if x1 < x2:
                    x1 -= difference[1]//2
                    x2 += difference[1]//2
                else:
                    x1 += difference[1]//2
                    x2 -= difference[1]//2
                if y1 < y2:
                    y1 -= difference[0]//2
                    y2 += difference[0]//2
                else:
                    y1 += difference[0]//2
                    y2 -= difference[0]//2

            # Append the coordinates to the list
            coordinates.append((class_id, x1, y1, x2, y2))
        return coordinates

def save_slices_from_ann(supervisely_path, img_path, output_path):
    global count
    total_images = len(os.listdir(img_path))
    progress_bar = tqdm(total=total_images, ncols=70)

    for image_filename in os.listdir(img_path):
        # Check if there is a corresponding annotation file
        ann_filename = image_filename.replace(".jpg", ".txt")
        ann_filename = ann_filename.replace(".png", ".txt")

        if not os.path.exists(os.path.join(supervisely_path, ann_filename)):
            print(f"Annotation not found: {ann_filename}")
            continue

        img = cv2.imread(os.path.join(img_path, image_filename))
        
        if img is None:
            print(f"Image not loaded: {image_filename}")
            continue

        if ann_filename.endswith(".txt"):
            coordinates = get_coordinates_from_txt(os.path.join(supervisely_path, ann_filename), img.shape)
            i = 0
            for class_id, x1, y1, x2, y2 in coordinates:
                
                # Check if the picture is larger than min_area pixels in area. If it is not, skip it
                if (x2 - x1) * (y2 - y1) < min_area or class_id > 1:
                    continue
                
                # Prime for saving images
                class_title = class_id_dict[int(class_id)]
                output_dir = os.path.join(output_path, class_title)
                # Create the output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)



                # Crop the rectangle from the image with a random offset such that the cones are not centered
                cropped_img = []
                cropped_img.append(random_shift(img, x1, y1, x2, y2, shift = False))
                if Extra:
                    for _ in range(3):
                        cropped_img.append(random_shift(img, x1, y1, x2, y2, shift = True))

                i += 1
               # Save the cropped rectangle
                for cropped_img_i in cropped_img:
                    
                    if cropped_img_i is not None:
                        # Define the output file name (you can modify this as needed)
                        output_filename = f"{class_title}_{image_filename}_crop{i}.png"

                        print(output_filename)

                        # Save the cropped rectangle
                        if not cv2.imwrite(os.path.join(output_dir, output_filename), cropped_img_i):
                            print(f"Failed to save image: {output_filename}")

                        count += 1


        progress_bar.update(1)

    progress_bar.close()

    print(f"Saved {count} images to {output_path}")

# Example usage:
save_slices_from_ann("Hog/YOLO_Dataset_All/labels/test", "Hog/YOLO_Dataset_All/images/test", "Hog/Slices_Fortest")
