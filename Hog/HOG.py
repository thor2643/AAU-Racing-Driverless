def calculate_HOG_features_custom(img, width = 64, height = 128):
    # To handle division by zero errors. This is a very small number, so it will not affect the result much
        epsilon = 1e-7
        # Convert the image to grayscale
        img = cv2.cvtColor(resize(img, width, height), cv2.COLOR_BGR2GRAY)
        img = np.float32(img) / 255.0

        # Calculate x and y gradients
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

        # Calculate magnitude and angle of gradient: g = sqrt(gx^2 + gy^2), angle = arctan(gy/gx)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # Create a histogram matrix
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

        # The histogram is 16 by 8. It is normalized in 16 by 16 pixels. It is rolled over the image by 8 pixels each time.
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
        for i in range(img.shape[0]//8 - 1): # There are 7 blocks in the x direction
            for j in range(img.shape[1]//8-1): # There are 15 blocks in the y direction
                for k in range(9): # There are 9 bins in the histogram
                    # There are 63 entires in each block, and 9 entries in each histogram 
                    feature_vector[i*63 + j*9 + k] = normalized_histogram[i, j, k]
        return feature_vector