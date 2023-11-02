# check if the numpy arrays in the file "Hog/HogFeatures/Yellow" are 3780 by 1

import numpy as np

# Load the a positive yellow sample
feature_vector1 = []
feature_vector2 = np.load("Hog/HogFeatures/Yellow/Yellow_cone_1.npy")

# stack the feature vectors
feature_vector = np.column_stack((feature_vector1, feature_vector2))

feature_vector3 = np.load("Hog/HogFeatures/Yellow/Yellow_cone_1.npy")

feature_vector = np.column_stack((feature_vector, feature_vector3))

print(feature_vector.shape)