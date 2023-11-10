import os
import yaml
import cv2

def save_slices_from_ann(supervisely_path, img_path, output_path):
    for filename, image in zip(os.listdir(supervisely_path), os.listdir(img_path)):
        img = cv2.imread(os.path.join(img_path, image))

        if filename.endswith(".json"):
            with open(os.path.join(supervisely_path, filename), 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)

                for obj in data['objects']:
                    x1, y1, x2, y2 = obj['points']['exterior'][0] + obj['points']['exterior'][1]

                    class_title = obj['classTitle']
                    output_dir = os.path.join(output_path, class_title)

                    # Create the output directory if it doesn't exist
                    os.makedirs(output_dir, exist_ok=True)

                    # Crop the rectangle from the image
                    cropped_img = img[y1:y2, x1:x2]

                    # Define the output file name (you can modify this as needed)
                    output_filename = f"{class_title}_{image}"

                    # Save the cropped rectangle
                    cv2.imwrite(os.path.join(output_dir, output_filename), cropped_img)

# Example usage:
save_slices_from_ann("Hog/DatafromThor/ann", "Hog/DatafromThor/img", "Hog/DatafromThor/slice")
