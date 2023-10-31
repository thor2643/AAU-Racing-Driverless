import numpy as np
import cv2

def read_frame(video_path, frame_number):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    print("video_path: "+video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return None

    # Read and discard frames until you reach the desired frame
    for _ in range(frame_number - 1):
        ret, _ = cap.read()
        if not ret:
            # End of the video or an error occurred
            print("Error: Could not read frame", frame_number)
            cap.release()
            return None

    # Read the desired frame
    ret, frame = cap.read()

    if not ret:
        # End of the video or an error occurred
        print("Error: Could not read frame", frame_number)
        cap.release()
        return None

    cap.release()
    return frame

def display_video_and_depth(video_path, depth_data_path, frame_number):
    # Read a frame from the video
    RGB_frame = read_frame(video_path, frame_number)

    if RGB_frame is None:
        print("Error: RGB_frame is empty.")
        return False

    # Read the depth data
    depth_data = np.load(depth_data_path)
    
    if depth_data is None:
        print("Error: Depth data is not loaded.")
        return False

    # Check that the dimensions of RGB_frame are not empty
    if RGB_frame.shape[0] == 0 or RGB_frame.shape[1] == 0:
        print("Error: Empty dimensions in RGB_frame.")
        return False

    # As a start, we don't want any values that are more than 4 meters. These values are set to 4000 mm. Threshold is set to 5000 mm
    Threshold = 2500
    depth_data[depth_data > Threshold] = Threshold

    # Normalize and map depth data to a grayscale image
    depth_data = 255 - (depth_data / Threshold * 255)
    depth_data = depth_data.astype(np.uint8)

    # Convert depth data to a 3-channel image
    depth_data_rgb = cv2.cvtColor(depth_data, cv2.COLOR_GRAY2BGR)

    # Resize the RGB frame to match the depth data dimensions
    RGB_frame = cv2.resize(RGB_frame, dsize=(depth_data.shape[1], depth_data.shape[0]), interpolation=cv2.INTER_AREA)

    # Stack the images horizontally
    images = np.hstack((RGB_frame, depth_data_rgb))

    # Display the image number to the side of the image
    cv2.putText(images, f"Frame {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Frame', images)

    # Wait for a keypress and store the key code
    cv2.waitKey(0)

    return True

def CountFrames(video_path = "Data_AccelerationTrack/1/Color.avi"):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
    else:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

        # Release the video capture object
        cap.release()
        return frame_count

# We start by making a function that can take a video, and the corrosponding npy array with the depth data, and the display them next to eachother, so we can see if the image makes sense to work on
def display_video_and_depth(video_path, path, depth_data_path, frame_number, Threshold = 5000):
    # Read a frame from the video
    RGB_frame = read_frame(path+video_path, frame_number)

    # Read the depth data
    depth_data = np.load(path + depth_data_path)
    
    # As a start, we don't want any values that are more than 4 meters. These values are set to 4000 mm. Threshold is set to 5000 mm
    depth_data[depth_data > Threshold] = Threshold

    # Normalize and map depth data to a grayscale image
    depth_data = 255 - (depth_data / Threshold * 255)
    depth_data = depth_data.astype(np.uint8)

    # Convert depth data to a 3-channel image
    depth_data_rgb = cv2.cvtColor(depth_data, cv2.COLOR_GRAY2BGR)

    # Resize the RGB frame to match the depth data dimensions
    RGB_frame = cv2.resize(RGB_frame, dsize=(depth_data.shape[1], depth_data.shape[0]), interpolation=cv2.INTER_AREA)
   
    # Stack the images horizontally
    images = np.hstack((RGB_frame, depth_data_rgb))

    # Display the image number to the side of the image
    cv2.putText(images, f"Frame {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Frame', images)

    # Wait for a keypress and store the key code
    cv2.waitKey(0)

    # Otherwise, continue to the next frame
    return True

def main(offset = 0, colorfile = "Color.avi", path = "Data_AccelerationTrack/4/"):
    length = CountFrames(path + colorfile)

    print("lenght: ", length)
    for i in range(1, length - offset):
        result = display_video_and_depth(colorfile, path, f"DepthData/depth_{i + offset}.npy", i + offset, 7000)
        print("i: ", i + offset)
        if not result:
            break

    # Close the window after all frames have been displayed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main(1560,"Color.avi")

