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

# Other tools
def SaveConesFromSlidingWindow(prediction, window):
    print(prediction)

    # Display the current window
    cv2.imshow("Window", window)

    # Wait for a key press to make a decision
    key = cv2.waitKey(0) & 0xFF

    if key == ord('s'):
        # Save the positive window to a file if 's' is pressed
        cv2.imwrite("Hog/Cones" + str(time.time()) + ".png", window)
    elif key == 27:  # Check for ESC key (ASCII value 27)
        return True  # Break out of the loop if ESC is pressed

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

def calculate_HOG_features_custom(img, width = 64, height = 128):
    # To handle division by zero errors. This is a very small number, so it will not affect the result much
        epsilon = 1e-7
        img = resize(img, width, height)

        img = np.float32(img) / 255.0
        
        # Calculate gradient
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        histogram = np.zeros((img.shape[0]//8, img.shape[1]//8, 9))
        normalized_histogram = np.zeros((img.shape[0]//8 - 1, img.shape[1]//8 - 1, 36))

        # Calculate Histogram of Gradients in 8×8 cells. Go trough each cell and calculate the histogram
        for i in range(img.shape[0]//8):
            for j in range(img.shape[1]//8):
                # Make a cutout of the cell
                cell_mag = mag[i*8:(i+1)*8, j*8:(j+1)*8]
                cell_angle = angle[i*8:(i+1)*8, j*8:(j+1)*8]

                # Convert all angle values above 180 to the same values below 180
                for k in range(8):
                    for l in range(8):
                        while cell_angle[k, l, 0] >= 180:
                            cell_angle[k, l, 0] -= 180

                # Calculate the histogram based on the magnitude and angle of the gradients
                hist = np.zeros(9)
                for k in range(8):
                    for l in range(8):

                        # Check what the angles is. If it is between 160 and 180, the value should porportionally be added to the 0 bin and the 160 bin
                        if cell_angle[k, l, 0] >= 160:
                            hist[0] += cell_mag[k, l, 0] * (180 - cell_angle[k, l, 0]) / 20
                            hist[8] += cell_mag[k, l, 0] * (cell_angle[k, l, 0] - 160) / 20
                            continue
                        else:
                            bin = int(cell_angle[k, l, 0] / 20)
                            hist[bin] += cell_mag[k, l, 0]     

                # Save the values in an array                

                histogram[i, j] = hist

        # The histogram is 16 by 8, we now normalize it in 16 by 16 pixels, which is cooresponding to 2 by 2 cells in this matrix.
        # We will roll this normalization over the entire image over the matrices of 2 by 2 cells by shifting it 1 cell the the side each time.
        for i in range(img.shape[0]//8 - 1):
            for j in range(img.shape[1]//8 - 1):
                # Normalize the histogram by making a 36 by 1 vector and normalizing it
                histogram_vector = np.zeros(36)
                histogram_vector[0:9] = histogram[i, j]
                histogram_vector[9:18] = histogram[i, j+1]
                histogram_vector[18:27] = histogram[i+1, j]
                histogram_vector[27:36] = histogram[i+1, j+1]
                histogram_vector = histogram_vector / (np.linalg.norm(histogram_vector) + epsilon)

                # Roll the normalized histogram back into the normalized histogram matrix
                normalized_histogram[i, j] = histogram_vector

        # Create the feature vector
        feature_vector = np.zeros(3780)
        for i in range(img.shape[0]//8 - 1):
            for j in range(img.shape[1]//8-1):
                for k in range(9):
                    feature_vector[i*63 + j*9 + k] = normalized_histogram[i, j, k]
        return feature_vector

def HOG_feature_extractor(input_image, useCustomHOG=False, target_width=64, target_height=128):
    if input_image is None:
        print("Failed to load or process the input image.")
        return None

    # Resize the image to 128x64
    input_image = np.array(resize(input_image, target_width, target_height))

    if not useCustomHOG:
        # Calculate the HOG features
        feature_vector = hog.compute(input_image)
    elif useCustomHOG:
        # Calculate the HOG features with your custom implementation
        feature_vector = calculate_HOG_features_custom(input_image, target_width, target_height)

    if feature_vector is None:
        print("Failed to calculate HOG features. Check the implementation of HOG feature extraction.")
    else:
        return feature_vector

def CalculateFeatureFromFolder(SamplesFolder,useCustomHOG,width,height):
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
            feature_vector = HOG_feature_extractor(target_image, useCustomHOG, width, height)

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

def HogFeatureFolderExtractor(PositiveSamplesFolder, NegativeSamplesFolder, useCustomHOG = True, width = 64, height = 128):
   
    # Process the positive and negative samples folders
    positive_features = CalculateFeatureFromFolder(PositiveSamplesFolder, useCustomHOG, width, height)
    negative_features = CalculateFeatureFromFolder(NegativeSamplesFolder, useCustomHOG, width, height)

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

def sliding_window(query_image, clf, custom_Hog = False, window_size = (64, 128), step_size = 2):
    # Iterate through the image using a sliding window
    cone_locations = []
    for y in range(0, query_image.shape[0] - window_size[1], step_size):
        for x in range(0, query_image.shape[1] - window_size[0], step_size):
            # Extract the current window
            window = query_image[y:y+window_size[1], x:x+window_size[0]]

            #Resize the window to 128x64
            window = resize(window, 64, 128)

            # Compute HOG features for the current window
            window_features = HOG_feature_extractor(window, custom_Hog, 64, 128)

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

    # Use the trained SVM to classify the window
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

def spot_check(whole_image, clf, location, custom_Hog=False, window_size=(64, 128), step_size=2, HighestID=0):
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
                window_features = HOG_feature_extractor(window, custom_Hog, 64, 128)

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
 
def HOG_predict(query_image, clf, custom_Hog=False, HighestID=0, step_factor=1, ):
    cone_locations = []

    Bias = 32

    # Split the image into three new images.
    top_middle = query_image[query_image.shape[0] // 2:query_image.shape[0] // 2 + Bias, 0:query_image.shape[1]]  # 32 x 16
    middle = query_image[query_image.shape[0] // 2:query_image.shape[0] // 2 + Bias * 3, 0:query_image.shape[1]]  # 64 x 32
    bottom = query_image[query_image.shape[0] // 2:query_image.shape[0], 0:query_image.shape[1]]  # 128 x 64

    # Define window sizes for each iteration
    window_sizes = [(16, 32), (32, 64), (64, 128)]

    # Iterate through the images and window sizes
    for s, sub_image in enumerate([top_middle, middle, bottom]):
        window_size = window_sizes[s]

        step_size = window_size[0] // 2// step_factor

        # Detect cones in the images
        locations = sliding_window(sub_image, clf, custom_Hog, window_size, step_size)
      
        # Transform the locations back to the original image
        for location in locations:
            (x, y), color = location

            # Append to the list with cones, while transforming the coordinates back to the original image
            cone_locations.append((HighestID, 0, (x, y + query_image.shape[0] // 2), color, window_size))

    # Filter out close points
    cone_locations = filter_close_points(cone_locations, 32)

    
    return cone_locations

def initialize_SVM_model(modelpath = "Hog/SVM_HOG_Model.pkl", PositiveSamplesFolder_y = "Hog/Cones_Positive/Centered/Yellow", PositiveSamplesFolder_b = "Hog/Cones_Positive/Centered/Blue", NegativeSamplesFolder = "Hog/Cones_Negative", kernel='rbf', C=1):
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

# Simulate racecar driving based on time
def simulate_racecar_driving(target_frame_rate=30):
    # Load video
    cap = cv2.VideoCapture("Data_AccelerationTrack/1/Color.avi")

    # Initialize the SVM models
    print("Initializing SVM model...")
    clf = initialize_SVM_model(modelpath="Hog/SVM_HOG_Model.pkl")
    KnownCones = []
    l = 0
    HighestID = 0
    while cap.isOpened():
        l += 1
        start_time = time.time()

        ret, frame = cap.read()
        if ret:
            # Detect cones in the frame
            cone_locations = HOG_predict(frame, clf, False, HighestID=HighestID)

            # If KnownCones is empty or not initialized correctly, initialize it as an empty list
            if not KnownCones or not isinstance(KnownCones, list):
                KnownCones = []

            # Track the cones
            KnownCones, HighestID = track_cones1(KnownCones, cone_locations, DistThreshold=64, clf=clf, image=frame, HighestID = HighestID)

            # Draw the cones on the frame
            for cone in KnownCones:
                # Ensure that the cone tuple is not empty before unpacking
                if cone:
                    cone_id, _, cone_location, color = cone

                    if color == "blue":
                        color = (255, 0, 0)
                    else:
                        color = (0, 255, 255)

                    #Check if the cone_location is a list or a tuple
                    if isinstance(cone_location[0], tuple):
                        cone_location = cone_location[0]

                    cv2.putText(frame, str(cone_id), (cone_location[0], cone_location[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                    cv2.rectangle(frame, cone_location, (cone_location[0] + 32, cone_location[1] + 64), color, 2)

            # Display the frame
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed_time = time.time() - start_time

            # Calculate the desired time to maintain the target frame rate
            desired_frame_time = 1 / target_frame_rate

            # If the processing time exceeds the desired frame time, skip frames
            while elapsed_time > desired_frame_time:
                ret, frame = cap.read()
                if not ret:
                    break
                elapsed_time -= desired_frame_time

            # If there's still time remaining, wait to maintain the frame rate
            if elapsed_time < desired_frame_time:
                time.sleep(desired_frame_time - elapsed_time)
        else:
            break

        if elapsed_time > 0:
            frame_rate = int(1 / elapsed_time)
            # print("Frame rate: " + str(frame_rate))

        print("Frame: " + str(l))

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

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

    # the first image in the test folder
    for images in os.listdir(Testpath_images):

        # Read the image
        img = cv2.imread(Testpath_images + images)
        # Read the Annotation file one line at a time

        Cones_from_ann = ReadAnnotationFile(img, images, Testpath_labels)

        # Detect cones in the frame
        cone_locations_HOG = HOG_predict(img, clf, False)

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

                if Iou >= 0.5:
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

        true_positives = close_state_hog.count(True)
        false_positives = close_state_hog.count(False)
        false_negatives = Close_state_ann.count(False)

        # We have chosen to set the precision to 0 if there are no true positives and no false positives as this is an undefinable case 
        if true_positives + false_positives == 0:
            Precision = 0
        elif (true_positives + false_negatives) == 0:
            Recall = 0
        else:
            Recall = true_positives/ (true_positives + false_negatives)
            Precision = true_positives / (true_positives + false_positives)   

        print(Recall)

        print("Recall: " + str(Recall))
        print("Precision: " + str(Precision))

        # Draw the found cones with blue  
        for cone in Cones_from_ann:
            if Close_state_ann[Cones_from_ann.index(cone)]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(img, (cone[0][0] - cone[1][0]//2 , cone[0][1] - cone[1][1]//2), (cone[0][0] + cone[1][0]//2, cone[0][1] + cone[1][1]//2), color, 2)         

        # Draw all the cones found 
        for cone in cone_locations_HOG:
            cv2.rectangle(img, (cone[2][0] - cone[4][0]//2, cone[2][1] - cone[4][1]//2), (cone[2][0] + cone[4][0]//2, cone[2][1] + cone[4][1]//2), (255, 0, 0), 2)

        # Display the frame - rezie the image to fit the screen
        img = cv2.resize(img, (1080, 720))

        cv2.imshow("Frame", img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        

    print("Testing logic...")

# Main logic
def main():
    # Load video
    cap = cv2.VideoCapture("Data_AccelerationTrack/1/Color.avi")

    # Initialize the SVM models
    print("Initializing SVM model...")
    clf = initialize_SVM_model(modelpath="Hog/SVM_HOG_Model.pkl")
    KnownCones = []
    l = 0
    HighestID = 0
    while cap.isOpened():
        l += 1
        print("frame" + str(l))

        ret, frame = cap.read()
        if ret:
            # Detect cones in the frame
            cone_locations = HOG_predict(frame, clf, False, HighestID=HighestID)

            # If KnownCones is empty or not initialized correctly, initialize it as an empty list
            if not KnownCones or not isinstance(KnownCones, list):
                KnownCones = []

            # Track the cones
            KnownCones, HighestID = track_cones1(KnownCones, cone_locations, DistThreshold=64, clf=clf, image=frame, HighestID = HighestID)

            # Draw the cones on the frame
            for cone in KnownCones:
                # Ensure that the cone tuple is not empty before unpacking
                if cone:
                    cone_id, _, cone_location, color = cone

                    if color == "blue":
                        color = (255, 0, 0)
                    else:
                        color = (0, 255, 255)

                    #Check if the cone_location is a list or a tuple
                    if isinstance(cone_location[0], tuple):
                        cone_location = cone_location[0]

                    cv2.putText(frame, str(cone_id), (cone_location[0], cone_location[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                    cv2.rectangle(frame, cone_location, (cone_location[0] + 32, cone_location[1] + 64), color, 2)

            # Display the frame
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break


    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #simulate_racecar_driving()
    #main()
    test_logic()
    print("Done")
