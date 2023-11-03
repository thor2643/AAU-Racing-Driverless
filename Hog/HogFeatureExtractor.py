import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Check if the image is 128x64, and if not resize it
def Resize(img):
    if img.shape[0] != 128 or img.shape[1] != 64:
        img = cv2.resize(img, (64, 128))
    return img

def featureExtraction(path):
    # To handle division by zero errors. This is a very small number, so it will not affect the result much
    epsilon = 1e-7

    # Read image and Resize image if it is not 128x64
    img = cv2.imread(path)
    img = Resize(img)

    # hog = cv2.HOGDescriptor()
    # hog_features = hog.compute(img)

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

                    # Check om vinklerne skal fordeles anderledes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

def main():
    img = cv2.imread("Hog/Cones/Bluecone_resized.jpg")
    img = Resize(img)
    
    featurevector = featureExtraction("Hog/Cones/Bluecone_resized.jpg")

    # Create a HOG descriptor object
    hog = cv2.HOGDescriptor()

    # Compute the HOG features
    features = hog.compute(img)

    # Visualize the features
    plt.plot(features)
    plt.plot(featurevector)
    plt.show()
    
def saveFeaturesInFolder(input_folder, output_folder):
    i = 0
    for filename in os.listdir(input_folder):
        # Extract the image name without extension
        # img_name = filename.split(".")[0]
        img_name = "NegativeSample" + str(i)

        feature_vector = featureExtraction(os.path.join(input_folder, filename))
        np.save(os.path.join(output_folder, img_name), feature_vector)
        i += 1  
    print("Done")

saveFeaturesInFolder("Hog/Cones/NegativeSamples", "Hog/HogFeatures/NegativeSamples")
