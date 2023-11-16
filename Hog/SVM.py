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
                distance = np.linalg.norm(np.array(current_point) - np.array(other_point))

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

def HOG_feature_extractor(input_image, useCustomHOG=True, target_width=64, target_height=128):
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

def sliding_window(query_image, clf, custom_Hog = False, window_size = (64, 128), step_size = 16):
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

            # Use the trained SVM to classify the window
            prediction = clf.predict([window_features])
            #print(prediction)

            # If the window is classified as a cone (1), mark it
            if prediction == 1:
                # Save the locations of the cones in an array
                cone_locations.append((x, y))

    return cone_locations

def spot_check(whole_image, clf, location, custom_Hog=False, window_size=(64, 128), step_size=16, HighestID=0):
    Check_kernel = [(-1, 1), (0, 1), (1, 1),
                    (-1, 0), (0, 0), (1, 0),
                    (-1, -1), (0, -1), (1, -1)]
    cone_locations = []

    # Turn around the coordinates for the location
    location = (location[1], location[0])

    # Iterate through the kernel
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

                # Use the trained SVM to classify the window
                prediction = clf.predict([window_features])

                # If the window is classified as a cone (1), mark it
                if prediction == 1:
                    # Save the locations of the cones in an array
                    cone_locations.append((HighestID +1, 0, (location[0] + x + x_offset, location[1] + y + y_offset)))
                    HighestID += 1

    return cone_locations   

def HOG_predict(query_image, clf, custom_Hog = False):
    # Iterate through the image using a sliding window
    cone_locations = []
    top_middle_cone_locations = []
    middle_cone_locations = []
    bottom_cone_locations = []

    Bias = 32

    #Split the image into three new images.
    top_middle = query_image[query_image.shape[0]//2:query_image.shape[0]//2 + Bias, 0:query_image.shape[1]]    # 32 x 16
    middle = query_image[query_image.shape[0]//2:query_image.shape[0]//2 + Bias * 3, 0:query_image.shape[1]]       # 64 x 32
    bottom = query_image[query_image.shape[0]//2:query_image.shape[0], 0:query_image.shape[1]]          # 128 x 64

    # Detect cones in the images
    top_middle_cone_locations = sliding_window(top_middle, clf, custom_Hog, (16, 32), 8)
    middle_cone_locations = sliding_window(middle, clf, custom_Hog, (32, 64), 16)
    bottom_cone_locations = sliding_window(bottom, clf, custom_Hog, (64, 128), 32)

    # Transform the locations back to the original image    
    for locations in [top_middle_cone_locations, middle_cone_locations, bottom_cone_locations]:
        for i in range(len(locations)):
            # Append to list with cones, while transforming the coordinates back to the original image
            cone_locations.append((locations[i][0], locations[i][1] + query_image.shape[0]//2))

    # Filter out close points
    cone_locations = filter_close_points(cone_locations, 32)

    return cone_locations

def initialize_SVM_model(modelpath = "Hog/SVM_HOG_Model.pkl", PositiveSamplesFolder = "Hog/Cones_Positive", NegativeSamplesFolder = "Hog/Cones_Negative", kernel='rbf', C=1):
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
        clf = train_SVM_model(PositiveSamplesFolder, NegativeSamplesFolder, kernel=kernel, C = C)

        # Save the model
        with open(modelpath, "wb") as f:
            pickle.dump(clf, f)
        print("Model was trained")

    return clf

# Object tracking
def update_cones(KnownCones, ConelocationsCurrent, DistThreshold=32, timeout_threshold=2, HighestID=0):
    updated_cones = []

    if KnownCones and ConelocationsCurrent:
        for i in range(len(KnownCones)):
            cone_updated = False

            if len(KnownCones[i]) >= 3:
                for j in range(len(ConelocationsCurrent)):
                    if ConelocationsCurrent[j]:
                        distance = np.linalg.norm(np.array(KnownCones[i][2]) - np.array(ConelocationsCurrent[j]))

                        if distance <= DistThreshold:
                            KnownCones[i] = (KnownCones[i][0], KnownCones[i][1] + 1, ConelocationsCurrent[j])
                            updated_cones.append(KnownCones[i])
                            cone_updated = True

                if not cone_updated:
                    KnownCones[i] = (KnownCones[i][0], KnownCones[i][1] + 1, KnownCones[i][2])
                    if KnownCones[i][1] <= timeout_threshold:
                        updated_cones.append(KnownCones[i])

    for cone_location in ConelocationsCurrent:
        if cone_location:
            cone_exists = any(np.array_equal(np.array(cone_location), np.array(cone[2])) for cone in KnownCones if len(cone) >= 3)
            if not cone_exists:
                HighestID += 1
                new_cone = (HighestID, 0, cone_location)
                updated_cones.append(new_cone)

    return updated_cones, HighestID

def track_cones1(KnownCones, ConelocationsCurrent, clf, image, DistThreshold=32, timeout_threshold=3, HighestID=0):
    print()
    print("KnownCones1:", KnownCones)

    Spot_cones = []
    for cone in KnownCones:
        if len(cone) >= 3:
            _, _, cone_location = cone
            Spot_cones.append(spot_check(image, clf, cone_location, False, (64, 128), 16, HighestID))

    # Extend the list of found cones, with the spotcheck of cones
    ConelocationsCurrent.extend(Spot_cones)

    # Update cones based on the current frame
    updated_cones, HighestID = update_cones(KnownCones, ConelocationsCurrent, DistThreshold, timeout_threshold, HighestID)

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
            cone_locations = HOG_predict(frame, clf, False)

            # If KnownCones is empty or not initialized correctly, initialize it as an empty list
            if not KnownCones or not isinstance(KnownCones, list):
                KnownCones = []

            # Track the cones
            KnownCones, HighestID = track_cones1(KnownCones, cone_locations, DistThreshold=64, clf=clf, image=frame, HighestID = HighestID)

            # Draw the cones on the frame
            for cone in KnownCones:
                # Ensure that the cone tuple is not empty before unpacking
                if cone:
                    cone_id, _, cone_location = cone
                    cv2.putText(frame, str(cone_id), (cone_location[0], cone_location[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, cone_location, (cone_location[0] + 32, cone_location[1] + 64), (0, 255, 0), 2)

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

# Main logic
def main():
    # Initialize the SVM model
    clf = initialize_SVM_model(C=0.001)

    # Load video
    cap = cv2.VideoCapture("Data_AccelerationTrack/1/Color.avi")

    # go trough the frames
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Detect cones in the frame
            cone_locations = HOG_predict(frame, clf, False)

            # Draw the cones on the frame
            for cone_location in cone_locations:
                cv2.rectangle(frame, cone_location, (cone_location[0] + 32, cone_location[1] + 64), (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()


# Main logic
def main():
    # Initialize the SVM model
    clf = initialize_SVM_model(C = 0.001)

    # Load video
    cap = cv2.VideoCapture("Data_AccelerationTrack/1/Color.avi")

    # go trough the frames
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Detect cones in the frame
            cone_locations = HOG_predict(frame, clf, False)

            # Draw the cones on the frame
            for cone_location in cone_locations:
                cv2.rectangle(frame, cone_location, (cone_location[0] + 32, cone_location[1] + 64), (0, 255, 0), 2)

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
    simulate_racecar_driving()
    #main()
    print("Done")
