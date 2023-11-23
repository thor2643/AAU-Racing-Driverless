import os
import random
import shutil


#--------------#--------------#---------------#--------------#--------------#

# Define path to data folder
input_image_path = "YOLO\\data\\images\\train"
input_label_path = "YOLO\\data\\labels\\train"

output_image_path = "YOLO\FullDataset\Images"
output_label_path = "YOLO\FullDataset\Labels"


# Define percentages for train, validation, and test data
train_percent = 0.7
val_percent = 0.2
test_percent = 0.1

#--------------#--------------#---------------#--------------#--------------#




# Create train, validation, and test folders
train_image_path = os.path.join(output_image_path, "train")
val_image_path = os.path.join(output_image_path, "val")
test_image_path = os.path.join(output_image_path, "test")

train_label_path = os.path.join(output_label_path, "train")
val_label_path = os.path.join(output_label_path, "val")
test_label_path = os.path.join(output_label_path, "test")


# Loop through files in data folder
for image, label in zip(os.listdir(input_image_path), os.listdir(input_label_path)):
    # Randomly assign file to train, validation, or test folder
    rand_num = random.random()

    if rand_num < train_percent:
        shutil.copy(os.path.join(input_image_path, image), train_image_path)
        shutil.copy(os.path.join(input_label_path, label), train_label_path)
    elif rand_num < train_percent + val_percent:
        shutil.copy(os.path.join(input_image_path, image), val_image_path)
        shutil.copy(os.path.join(input_label_path, label), val_label_path)
    else:
        shutil.copy(os.path.join(input_image_path, image), test_image_path)
        shutil.copy(os.path.join(input_label_path, label), test_label_path)

