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


def filter_close_points(cone_locations, distance_threshold):
    filtered_cone_locations = []
    ignore_flags = []

    for i, current_point in enumerate(cone_locations):
        if i in ignore_flags:
            continue  # Skip points marked for ignoring

        keep_point = True

        for j, other_point in enumerate(cone_locations):
            if i != j and j not in ignore_flags:  # Skip comparing the point with itself and with already ignored points
                distance = np.linalg.norm(np.array(current_point[2]) - np.array(other_point[2]))

                # Check if the distance is below the threshold
                if distance < distance_threshold:
                    keep_point = False
                    ignore_flags.append(j) # Mark the other point for ignoring
                    break 
        if keep_point:
            filtered_cone_locations.append(current_point)

    return filtered_cone_locations

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

def HogFeatureFolderExtractor(PositiveSamplesFolder, NegativeSamplesFolder, width = 64, height = 128):
   
    # Process the positive and negative samples folders
    positive_features = CalculateFeatureFromFolder(PositiveSamplesFolder, width, height)
    negative_features = CalculateFeatureFromFolder(NegativeSamplesFolder, width, height)

    # Transpose the feature vectors
    positive_features = positive_features.T
    negative_features = negative_features.T

    # Combine features and create labels for positive and negative samples
    positive_labels = np.ones(len(positive_features))
    negative_labels = np.zeros(len(negative_features))

    return positive_features, positive_labels, negative_features, negative_labels

# SVM classifier and prediction
def train_SVM_model(PositiveSamplesFolder, NegativeSamplesFolder, kernel='linear', C=1):
    positive_features, positive_labels, negative_features, negative_labels, = HogFeatureFolderExtractor(PositiveSamplesFolder, NegativeSamplesFolder, False, width= 64, height = 128)

    # Combine all features and labels
    x = np.vstack((positive_features, negative_features))
    y = np.concatenate((positive_labels, negative_labels))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train an SVM classifier - Kernel is the kernel type, C is the penalty parameter of the error term
    clf = svm.SVC(kernel=kernel, C=C, probability=True)
    clf.fit(X_train, y_train)

    # Calculate the accuracy of the classifier
    y_pred = clf.predict(X_test)
    print("Accuracy: " + str(accuracy_score(y_test, y_pred)))

    return clf

def sliding_window(query_image, clf, window_size = (64, 128), step_size = 8):
    # Iterate through the image using a sliding window
    cone_locations = []
    print("image size:" + str(query_image.shape))
    Stop1 = query_image.shape[0] - window_size[1]
    Stop2 = query_image.shape[1] - window_size[0]

    for y in range(0, Stop1, step_size):
        for x in range(0, Stop2, step_size):
            # Extract the current window
            window = query_image[y:y+window_size[1], x:x+window_size[0]]

            #Resize the window to 128x64
            window = resize(window, 64, 128)

            # Compute HOG features for the current window
            window_features = HOG_feature_extractor(window, 64, 128)

            c = Predict(clf, window_features, x ,y)

            if c:
                cone_locations.extend(c)

    return cone_locations

def Predict(clf, window, x, y):
    # Check if the received input is valid
    if window is None:
         raise TypeError("Window is None")
   # Check if x and y are integers
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError("x and y must be integers")

    cone_locations = []

    # Use the trained SVM to classify the window - SVM's are not probabilistic classifiers, so we use the function to get the label
    # We can find the probability of the prediction using the probability function, but this will use 5-fold cross validation, and will surely slow down the process
    prediction_y = clf[1].predict([window])[0]
    prediction_b = clf[0].predict([window])[0]

    # If the window is classified as a cone (1), mark it
    if prediction_y == 1 and prediction_b == 1:
        # Save the locations of the cones in an array
        cone_locations.append(((x, y), "blue"))
        #print("Blue cone found")
    elif prediction_b == 1:
        #print("Blue cone found")
        cone_locations.append(((x, y), "blue"))       
    elif prediction_y == 1:
        #print("Yellow cone found")
        cone_locations.append(((x, y), "yellow"))
    
    return cone_locations

def spot_check(whole_image, clf, location, window_size=(64, 128), step_size=2, HighestID=0):
    Check_kernel = [(-1, 1), (0, 1), (1, 1),
                    (-1, 0), (0, 0), (1, 0),
                    (-1, -1), (0, -1), (1, -1)]
    cone_locations = []

    # Turn around the coordinates for the location
    location = (location[1], location[0])

    for i in range(len(Check_kernel)):
        x_offset = step_size * Check_kernel[i][0]
        y_offset = step_size * Check_kernel[i][1]

        # Define the region of interest (ROI)
        roi = whole_image[location[0] + x_offset:location[0] + window_size[1] + x_offset,
                  location[1] + y_offset:location[1] + window_size[0] + y_offset]

        # Iterate through the ROI using sliding window technique
        for x in range(0, roi.shape[0] - window_size[1] + 1, step_size):
            for y in range(0, roi.shape[1] - window_size[0] + 1, step_size):
                window = roi[x:x + window_size[1], y:y + window_size[0]]

                # Print window dimensions and check if it's not empty
                if window.size == 0:
                    print("Error: Window is empty!")
                    continue

                # Resize the window to 128x64
                window = resize(window, 64, 128)

                # Compute HOG features for the current window
                window_features = HOG_feature_extractor(window, 64, 128)

                cone_location = Predict(clf, window_features, x, y)
                if cone_location and x + x_offset >= 0 and y + y_offset >= 0:
                    # Ensure that the list has the correct structure ((x, y), color)
                    if len(cone_location[0]) != 2:
                        print("Error: Cone location is not correct")
                        continue
                    
                    # Save the locations of the cones in an array
                    cone_locations.append((HighestID, 0, (x + x_offset, y + y_offset), cone_location[0][1]))
                    HighestID += 1

    return cone_locations, HighestID
 
def HOG_predict(query_image, clf, HighestID=0, step_factor=1, Simple_Slide = True):
    cone_locations = []

    Bias = 32
    # Define window sizes for each iteration
    window_sizes = [(16, 32), (32, 64), (64, 128)]

    if Simple_Slide:
         # Iterate through the images and window sizes
        for s, _ in enumerate(window_sizes):
            print(s)
            window_size = window_sizes[s]
            print(window_size)

            step_size = window_size[0] // 2// step_factor
            

            # Detect cones in the images
            locations = sliding_window(query_image, clf, window_size, step_size)
        
            for location in locations:
                (x, y), color = location

                # Append to the list with cones, while transforming the coordinates back to the original image
                cone_locations.append((HighestID, 0, (x, y), color, window_size))
    else:
        # Split the image into three new images.
        top_middle = query_image[query_image.shape[0] // 2:query_image.shape[0] // 2 + Bias, 0:query_image.shape[1]]  # 32 x 16
        middle = query_image[query_image.shape[0] // 2:query_image.shape[0] // 2 + Bias * 3, 0:query_image.shape[1]]  # 64 x 32
        bottom = query_image[query_image.shape[0] // 2:query_image.shape[0], 0:query_image.shape[1]]  # 128 x 64

        # Iterate through the images and window sizes
        for s, sub_image in enumerate([top_middle, middle, bottom]):
            window_size = window_sizes[s]

            step_size = window_size[0]// step_factor
            print("Window:" + str(s) + ", stepsize: " + str(step_size))

            # Detect cones in the images
            locations = sliding_window(sub_image, clf, window_size, step_size)
        
            # Transform the locations back to the original image
            for location in locations:
                (x, y), color = location

                # Append to the list with cones, while transforming the coordinates back to the original image
                cone_locations.append((HighestID, 0, (x, y + query_image.shape[0] // 2), color, window_size))

    # Filter out close points
    cone_locations = filter_close_points(cone_locations, 6)

    
    return cone_locations

def initialize_SVM_model(modelpath = "Hog/SVM_HOG_Model.pkl", PositiveSamplesFolder_y = "Hog/Slices/Yellow", PositiveSamplesFolder_b = "Hog/Slices/Blue", NegativeSamplesFolder = "Hog/Slices/Negative_samples", kernel='rbf', C=1):
    # First we check if we have a model trained already, if not we should train one
    print("Checking if a model is already trained...")
    if os.path.isfile(modelpath):
        # Load the model
        with open(modelpath, "rb") as f:
            clf = pickle.load(f)
            print("Model was loaded with the given parameters: " + str(clf))
    else:
        print("No model was found. Training a new model...")
        # Train the model
        print("Training the model for the blue cones...")
        clf_b = train_SVM_model(PositiveSamplesFolder_b, NegativeSamplesFolder, kernel=kernel, C = C)
        print("Training the model for the yellow cones...")
        clf_y = train_SVM_model(PositiveSamplesFolder_y, NegativeSamplesFolder, kernel=kernel, C = C)
        clf = [clf_b, clf_y]

        # Save the model
        with open(modelpath, "wb") as f:
            pickle.dump(clf, f)
        print("Model was trained")

    return clf

# Object tracking
def update_cones(KnownCones, ConelocationsCurrent, DistThreshold=64, timeout_threshold=3, HighestID=0):
    updated_cones = []

    if not isinstance(KnownCones, list):
        KnownCones = []

    if KnownCones and ConelocationsCurrent:
        for i, known_cone in enumerate(KnownCones):
            cone_updated = False
             
            if isinstance(known_cone, tuple) and len(known_cone) >= 3:
                for cone_id, update_variable, (x, y), color in ConelocationsCurrent:
                    if cone_id is not None and (x, y):
                        distance = np.linalg.norm(np.array(known_cone[2][:2]) - np.array((x, y)))
                        if distance <= DistThreshold:
                            KnownCones[i] = (KnownCones[i][0], 0, (x, y), color)
                            updated_cones.append(KnownCones[i])
                            cone_updated = True
                            break  # Exit the loop after updating a cone

                if not cone_updated:
                    KnownCones[i] = (KnownCones[i][0], KnownCones[i][1] + 1, KnownCones[i][2], KnownCones[i][3])
                    if KnownCones[i][1] <= timeout_threshold:
                        updated_cones.append(KnownCones[i])
    elif KnownCones:
        for i, known_cone in enumerate(KnownCones):
            # Decrement the timeout counter or take other appropriate action
            KnownCones[i] = (KnownCones[i][0], KnownCones[i][1] + 1, KnownCones[i][2], KnownCones[i][3])

            # Check if the timeout counter is still within acceptable limits
            if KnownCones[i][1] <= timeout_threshold:
                updated_cones.append(KnownCones[i])

    for cone_id, update_variable, (x, y), color in ConelocationsCurrent:
        if cone_id is not None and (x, y):
            # If KnownCones is empty or not initialized correctly, initialize it as an empty list
            if not KnownCones or not isinstance(KnownCones, list):
                KnownCones = []

            # Ensure cone_location has at least one element before attempting to access its elements
            # Check if there are any cones in KnownCones
            if KnownCones:
                # Check if a similar cone already exists in KnownCones
                cone_exists = any(
                    np.linalg.norm(np.array((x, y)) - np.array(cone[2][:2])) <= DistThreshold
                    for cone in KnownCones if isinstance(cone, tuple) and len(cone) >= 3
                )

                # If a similar cone doesn't exist, add a new cone to updated_cones
                if not cone_exists:
                    HighestID += 1
                    new_cone = (HighestID, 0, (x, y), color)
                    updated_cones.append(new_cone)
            else:
                # KnownCones is empty, so add a new cone to updated_cones
                HighestID += 1
                new_cone = (HighestID, 0, (x, y), color)
                updated_cones.append(new_cone)

    return updated_cones, HighestID

def track_cones1(KnownCones, ConelocationsCurrent, clf, image, DistThreshold=32, timeout_threshold=3, HighestID=0):
    Spot_cones = []
    for cone in KnownCones:
        if len(cone) >= 3:
            _, _, cone_location, color = cone
            cones_to_add, HighestID = spot_check(image, clf, cone_location, False, (64, 128), 16, HighestID)
            if cones_to_add:
                Spot_cones.append(cone)

    # Extend the list of found cones, with the spotcheck of cones
    if Spot_cones:
        ConelocationsCurrent.extend(Spot_cones)

    print("Known cones before: " + str(KnownCones))
    print("Found cones: " + str(ConelocationsCurrent))
    # Update cones based on the current frame
    updated_cones, HighestID = update_cones(KnownCones, ConelocationsCurrent, DistThreshold, timeout_threshold, HighestID)
    print("Known cones after: " + str(updated_cones))

    return updated_cones, HighestID

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

def IOU(boxA, boxB):
    # Extract the coordinates of the boxes
    x0A, y0A, x1A, y1A = boxA
    x0B, y0B, x1B, y1B = boxB
    
    # Determine the (x, y)-coordinates of the intersection rectangle
    l_x = max(x0A, x0B)
    r_x = min(x1A, x1B)
    t_y = max(y0A, y0B)
    b_y = min(y1A, y1B)

    # Compute the area of intersection rectangle
    interArea = max(0, r_x - l_x) * max(0, b_y - t_y)
    
    # If the area is non-positive, the boxes don't intersect
    if interArea <= 0:
        Iou = 0
        return Iou

    # Compute the area of both rectangles
    area_box_a = abs(x1A - x0A) * abs(y1A - y0A)
    area_box_b = abs(x1B - x0B) * abs(y1B - y0B)

    # Compute the intersection over union
    Union = area_box_a + area_box_b - interArea

    if interArea > area_box_a + area_box_b:
        print("Error: wtf")

    # Compute the intersection over union
    Iou = interArea / Union

    #print("Iou: " + str(Iou))
    return Iou
 
# Test Logic
def test_logic(Testpath_images = "Hog/Test/images/", Testpath_labels = "Hog/Test/label/"):
    # Load the SVM model
    clf = initialize_SVM_model(modelpath="Hog/SVM_HOG_Model.pkl")
    true_positives = 0
    false_positives = 0 
    false_negatives = 0 
    Recall = 0
    Precision = 0

    # Make a progress bar
    pbar = tqdm(total=len(os.listdir(Testpath_images)), desc="Processing images", unit="image")

    # Iterate through all images in the folder
    for images in os.listdir(Testpath_images):
        # Read the image
        img = cv2.imread(Testpath_images + images)
        # Read the Annotation file one line at a time

        Cones_from_ann = ReadAnnotationFile(img, images, Testpath_labels)
        
        # Detect cones in the frame
        cone_locations_HOG = HOG_predict(img, clf, False, step_factor= 4, Simple_Slide = False)

        # Initiate the state of the cones as the lenght of the cones from the annotation file
        Close_state_ann = len(Cones_from_ann) * [False]
        close_state_hog = len(cone_locations_HOG) * [False]

        # Run a intersection over union check to see if the cones are close to each other - THIS IS OLD CODE
        for cone in cone_locations_HOG:
            close_cones = []
            for i, cone_from_ann in enumerate(Cones_from_ann):
                # Extract the coordinates of the boxes

                # Extracting coordinates for cone A
                x0A = max(cone[2][0] - cone[4][0] // 2, 0)
                y0A = max(cone[2][1] - cone[4][1] // 2, 0)
                x1A = max(cone[2][0] + cone[4][0] // 2, 0)
                y1A = max(cone[2][1] + cone[4][1] // 2, 0)

                # Extracting coordinates for cone B
                x0B = max(cone_from_ann[0][0] - cone_from_ann[1][0] // 2, 0)
                y0B = max(cone_from_ann[0][1] - cone_from_ann[1][1] // 2, 0)
                x1B = max(cone_from_ann[0][0] + cone_from_ann[1][0] // 2, 0)
                y1B = max(cone_from_ann[0][1] + cone_from_ann[1][1] // 2, 0)
                

                # Calculate the intersection over union
                Iou = IOU((x0A, y0A, x1A, y1A), (x0B, y0B, x1B, y1B))

                if Iou >= 0.18:
                    # If the cones are close to each other, save the index, and the IOU value. Only the closest cone will be saved
                    close_cones.append((i, Iou))

            # If there are any close cones, save the closest one
            if close_cones:
                # Mark hog cone as found
                close_state_hog[cone_locations_HOG.index(cone)] = True

                # Sort the list of close cones by IOU value
                close_cones.sort(key=lambda x: x[1], reverse=True)

                # Save the index of the closest cone
                Close_state_ann[close_cones[0][0]] = True    

        # Draw the cones from the annotation file
        for cone in Cones_from_ann:
            cv2.rectangle(img, (cone[0][0] - cone[1][0] // 2, cone[0][1] - cone[1][1] // 2), (cone[0][0] + cone[1][0] // 2, cone[0][1] + cone[1][1] // 2), (255, 0, 0), 2)

        # Draw the cones from the HOG detector, make them green if they are flagged as close to a cone from the annotation file
        for i, cone in enumerate(cone_locations_HOG):
            if close_state_hog[i]:
                cv2.rectangle(img, (cone[2][0] - cone[4][0] // 2, cone[2][1] - cone[4][1] // 2), (cone[2][0] + cone[4][0] // 2, cone[2][1] + cone[4][1] // 2), (0, 255, 0), 2)
            else:
                cv2.rectangle(img, (cone[2][0] - cone[4][0] // 2, cone[2][1] - cone[4][1] // 2), (cone[2][0] + cone[4][0] // 2, cone[2][1] + cone[4][1] // 2), (0, 0, 255), 2)
        
        # Save the image in a folder
        cv2.imwrite("Hog/TestImagesDrawn" + images, img)

        # Calculate the true positives, false positives and false negatives - using the count function
        true_positives += Close_state_ann.count(True)
        false_negatives += Close_state_ann.count(False)
        false_positives += close_state_hog.count(False)    
        
        if true_positives + false_positives == 0:
            Precision = 0
        elif (true_positives + false_negatives) == 0:
            Recall = 0
        else:
            Recall = true_positives/ (true_positives + false_negatives)
            Precision = true_positives / (true_positives + false_positives)   

        # Update progress bar
        pbar.update(1)

        print("True positives: " + str(true_positives))
        print("False positives: " + str(false_positives))
        print("False negatives: " + str(false_negatives))
        print("Precision: " + str(Precision))
        print("Recall: " + str(Recall)) 

    # Close the progress bar
    pbar.close()


test_logic()