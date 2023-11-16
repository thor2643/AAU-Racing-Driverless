import cv2
import os
import time
index = 0

# Function to save cropped images based on sliding window
def save_cropped_image(window, output_folder="Hog", prefix="Cones"):
    global index  # Declare index as a global variable

    # Save the positive window to a file if 's' is pressed
    timestamp = int(time.time())
    file_path = os.path.join(output_folder, f"{prefix}_{timestamp}_{index}.png")
    cv2.imwrite(file_path, window)
    index += 1
    print(f"Image saved: {file_path}")


# Function to apply sliding window
def process_frames_and_save(video_path, output_folder="Cones", window_size=(64, 128), step_size=16):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    wait_key = 0      # Variable to store the result of cv2.waitKey

    Bias = 32

    while True:
        # Read the frame
        ret, frame = cap.read()

        if not ret:
            break

        # Display the frame
        cv2.imshow("Frame", frame)

        #Split the image into three new images.
        top_middle = frame[frame.shape[0]//2:frame.shape[0]//2 + Bias, 0:frame.shape[1]]    # 32 x 16
        middle = frame[frame.shape[0]//2:frame.shape[0]//2 + Bias * 3, 0:frame.shape[1]]       # 64 x 32
        bottom = frame[frame.shape[0]//2:frame.shape[0], 0:frame.shape[1]]          # 128 x 64

        # Detect cones in the images
        sliding_window(top_middle, (16, 32), 8)
        sliding_window(middle, (32, 64), 16)
        sliding_window(bottom, (64, 128), 32)

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

def sliding_window(query_image, window_size=(64, 128), step_size=16):
    # Iterate through the image using a sliding window
    cone_locations = []
    for y in range(0, query_image.shape[0] - window_size[1], step_size):
        for x in range(0, query_image.shape[1] - window_size[0], step_size):
            # Extract the window from the frame
            window = query_image[y:y + window_size[1], x:x + window_size[0]]

            # Display the window
            cv2.imshow("Window", window)

            # Wait for a key press
            wait_key = cv2.waitKey(0) & 0xFF

            # Save the positive window to a file if 's' is pressed
            if wait_key == ord('s'):
                index = save_cropped_image(window)
                cone_locations.append((x, y))
            # Move to the next frame if 'k' is pressed
            elif wait_key == ord('k'):
                # Break out of the function
                return
            elif wait_key == ord('j'):
                # Continue to the next window
                continue
                







    return cone_locations

# ... (rest of the code remains unchanged)

if __name__ == "__main__":
    # Specify the video path
    video_path = "Data_AccelerationTrack/3/Color.avi"

    # Process frames and save based on sliding window
    process_frames_and_save(video_path)
    print("Done")
