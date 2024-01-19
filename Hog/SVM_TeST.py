import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle
import time
from tqdm import tqdm

hog = cv2.HOGDescriptor()

#HOG feature extractor
def resize(img, width = 64, height = 128):
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height))
    return img

def HOG_feature_extractor(input_image, target_width=64, target_height=128):
    if input_image is None:
        print("Failed to load or process the input image.")
        return None

    # Resize the image to 128x64
    input_image = np.array(resize(input_image, target_width, target_height))

    # Calculate the HOG features
    feature_vector = hog.compute(input_image)
 
    if feature_vector is None:
        print("Failed to calculate HOG features. Check the implementation of HOG feature extraction.")
    else:
        return feature_vector

def CalculateFeatureFromFolder(SamplesFolder, width,height):
    features = []
    correcly_loaded = 0
    total_load_attempts = 0

    # Iterate through all subfolders and files
    for root, dirs, files in os.walk(SamplesFolder):
        for filename in tqdm(files, desc="Processing files", unit="file"):
            file_path = os.path.join(root, filename)
            
            # Skip non-image files
            if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Read the image
            target_image = cv2.imread(file_path)

            # Extract HOG features
            feature_vector = HOG_feature_extractor(target_image, width, height)

            # Check if the feature vector is valid
            if not (np.isnan(feature_vector).any()) and (feature_vector.shape[0] == 3780):
                if features is None:
                    features = [feature_vector]
                else:
                    correcly_loaded += 1
                    features.append(feature_vector)

            total_load_attempts += 1

    # Check how many features are loaded correctly out of the total number of features
    print("Number of features loaded correctly: " + str(correcly_loaded) + "/" + str(total_load_attempts))

    # Convert the features list to a numpy array
    features = np.array(features).T
    print(features.shape)

    return features

# SVM classifier and prediction


def Predict(clf, window):
    # Check if the received input is valid
    if window is None:
         raise TypeError("Window is None")

    # Calculate the HOG features
    window = HOG_feature_extractor(window, 64, 128)

    # Use the trained SVM to classify the window - SVM's are not probabilistic classifiers, so we use the function to get the label
    # We can find the probability of the prediction using the probability function, but this will use 5-fold cross validation, and will surely slow down the process
    prediction_y = clf[1].predict([window])[0]
    prediction_b = clf[0].predict([window])[0]

    # If the window is classified as a cone (1), mark it
    if prediction_y == 1 and prediction_b == 1:
        return "Yellow"
    elif prediction_b == 1:
        return "Blue"  
    elif prediction_y == 1:
        return "Yellow"
    else:
        return "Negative"

def initialize_SVM_model(modelpath="Hog/SVM_HOG_Model.pkl"):
    # First we check if we have a model trained already, if not we should train one
    print("Checking if a model is already trained...")
    if os.path.isfile(modelpath):
        # Load the model
        with open(modelpath, "rb") as f:
            clf = pickle.load(f)
            print("Model was loaded with the given parameters: " + str(clf))
            return clf  # Add this line to return the loaded classifier
    else:
        print("No model was found. Training a new model...")

        # Stop program if no model is found
        raise NotADirectoryError("No model was found. Please train a model first.")

def ReadAnnotationFile(img, image_name, Testpath_labels):
    with open(Testpath_labels + image_name[:-4] + ".txt") as f:
        Cones = []
        for line in f:
            # Split the line into a list
            line = line.split()

            # Interpret color
            if line[0] == "0":
                color = "yellow"
            elif line[0] == "1":
                color = "blue"
            else:
                # Skip the current line if the color is not recognized
                continue
                
            x = int(float(line[1]) * img.shape[1])
            y = int(float(line[2]) * img.shape[0])
            w = int(float(line[3]) * img.shape[1])
            h = int(float(line[4]) * img.shape[0])

            # Extract the cone location and color from the list
            Cone_location = [(x,y), (w,h), color]  
            
            Cones.append(Cone_location)   

    return Cones 

# Test Logic
def test_logic(Testpath_images = "Hog/Slices_test/", Testpath_labels = "Hog/Test/label/"):
    # Load the SVM model
    clf = initialize_SVM_model(modelpath="Hog/SVM_HOG_Model.pkl")
    
    # Initialize the metrics - True positives, false positives, false negatives
    Metric_b = [0,0,0]
    Metric_y = [0,0,0]
    Recall = 0
    false_positives_n= [0,0]
    Precision = 0


    # Make a progress bar
    pbar = tqdm(total=len(os.listdir(Testpath_images)), desc="Processing images", unit="image")

    # Iterate through all images in the sub folders
    folders = ["Blue", "Yellow", "Negative_samples"]

    # Iterate through all folders
    for folder in folders:
        folder_path = os.path.join(Testpath_images, folder)

        # Iterate through all images in the folder
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                # Read the image
                img = cv2.imread(os.path.join(root, filename))


                prediction = Predict(clf, img)

                # Based on the folder check if the prediction is correct
                if folder == "Blue":
                    if prediction == "Blue":
                        Metric_b[0] += 1
                    elif prediction == "Yellow":
                        Metric_b[1] += 1
                    else:
                        Metric_b[2] +=1
                elif folder == "Yellow":
                    if prediction == "Yellow":
                        Metric_y[0] += 1
                    elif prediction == "Blue":
                        Metric_y[1] += 1
                    else:
                        Metric_y[2] +=1
                elif folder == "Negative_samples":
                    if prediction == "Blue":
                        false_positives_n[0] += 1
                    elif prediction == "Yellow":
                        false_positives_n[1] += 1

                # Update progress bar
                pbar.update(1)
    
    # Close the progress bar
    pbar.close()

    # Calculate the metrics
    true_positives = Metric_b[0] + Metric_y[0]
    false_positives = Metric_b[1] + Metric_y[1] + false_positives_n[0] + false_positives_n[1]
    false_negatives = Metric_b[2] + Metric_y[2]

    # We have chosen to set the precision to 0 if there are no true positives and no false positives as this is an undefinable case 
    if true_positives + false_positives == 0:
        Precision = 0
    elif (true_positives + false_negatives) == 0:
        Recall = 0
    else:
        Recall = true_positives/ (true_positives + false_negatives)
        Precision = true_positives / (true_positives + false_positives)   

    print("True positives: " + str(true_positives))
    print("False positives: " + str(false_positives))
    print("False negatives: " + str(false_negatives))
    print("False positives from blue model: " + str(Metric_y[1]))
    print("False positives from yellow model: " + str(Metric_b[1]))
    print("False positives from negative samples: " + str(false_positives_n))
    print("Precision: " + str(Precision))
    print("Recall: " + str(Recall)) 

test_logic()