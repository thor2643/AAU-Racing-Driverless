import os
import yaml
import cv2
import numpy as np

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

def save_slices_from_ann(supervisely_path, img_path, output_path):
    for filename, image_filename in zip(os.listdir(supervisely_path), os.listdir(img_path)):
        img = cv2.imread(os.path.join(img_path, image_filename))

        if filename.endswith(".json"):
            with open(os.path.join(supervisely_path, filename), 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)

                for obj in data['objects']:
                    x1, y1, x2, y2 = obj['points']['exterior'][0] + obj['points']['exterior'][1]

                    class_title = obj['classTitle']
                    output_dir = os.path.join(output_path, class_title)

                    # Create the output directory if it doesn't exist
                    os.makedirs(output_dir, exist_ok=True)

                    # Crop the rectangle from the image with a random offset such that the cones are not centered
                    cropped_img = random_shift(img, x1, y1, x2, y2, 0)

                    # Check if the cropped image is not None and not empty before saving
                    if cropped_img is not None and not cropped_img.size == 0:
                        # Define the output file name (you can modify this as needed)
                        output_filename = f"{class_title}_{image_filename}"

                        # Save the cropped rectangle
                        cv2.imwrite(os.path.join(output_dir, output_filename), cropped_img)

# Example usage:
save_slices_from_ann("Hog/amz/ann", "Hog/amz/img", "Hog/Slice")
