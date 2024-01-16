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

min_area = 400

def random_shift(image, x1, y1, x2, y2, max_shift=20):
    # Calculate random shifts in both x and y directions
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    shift_y = np.random.randint(-max_shift, max_shift + 1)

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
            # Append the coordinates to the list
            coordinates.append((class_id, x1, y1, x2, y2))
        return coordinates

def save_slices_from_ann(supervisely_path, img_path, output_path):
    total_images = len(os.listdir(img_path))
    progress_bar = tqdm(total=total_images, ncols=70)

    for filename, image_filename in zip(os.listdir(supervisely_path), os.listdir(img_path)):
        img = cv2.imread(os.path.join(img_path, image_filename))
        if img is None:
            print(f"Image not loaded: {image_filename}")
            continue


        if filename.endswith(".txt"):
            coordinates = get_coordinates_from_txt(os.path.join(supervisely_path, filename), img.shape)
            for class_id, x1, y1, x2, y2 in coordinates:
                

                # Check if the picture is larger than 200 pixels in area. If it is not, skip it
                if (x2 - x1) * (y2 - y1) < min_area or class_id > 1:
                    continue

                class_title = class_id_dict[int(class_id)]
                output_dir = os.path.join(output_path, class_title)

                # Create the output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)

                # Crop the rectangle from the image with a random offset such that the cones are not centered
                cropped_img = random_shift(img, x1, y1, x2, y2, 0)
                if cropped_img is None:
                    print(f"Invalid crop for image: {image_filename}")
                    continue


                # Check if the cropped image is not None and not empty before saving
                if cropped_img is not None and not cropped_img.size == 0:
                    # Define the output file name (you can modify this as needed)
                    output_filename = f"{class_title}_{image_filename}"

                    # Save the cropped rectangle
                    if not cv2.imwrite(os.path.join(output_dir, output_filename), cropped_img):
                        print(f"Image not saved: {output_filename}")

        progress_bar.update(1)

    progress_bar.close()


# Example usage:
save_slices_from_ann("Hog/YOLO_Dataset_All/labels/train", "Hog/YOLO_Dataset_All/images/train", "Hog/Slices")
