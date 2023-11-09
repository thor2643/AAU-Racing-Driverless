import pyzed.sl as sl
import os
import numpy as np
import cv2

# Create a ZED camera object
zed = sl.Camera()
# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
frame_rate = 30
init_params.camera_fps = frame_rate

image = sl.Mat()
point_cloud = sl.Mat()

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS) : #Ensure the camera has opened succesfully
    exit(-1)

# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()

# Set custom output file names if provided, or use defaults
color_output_file = "ZED_color_video.avi"
codec = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(color_output_file, codec, frame_rate, (1280, 720), True)

# Create the "DepthData" directory if it doesn't exist
if not os.path.exists("DepthData"):
    os.makedirs("DepthData")

frame_counter = 0  # Initialize frame counter


while True:
    # A new image is available if grab() returns SUCCESS
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        # Get the numpy array from the sl.Mat object
        image_np = cv2.cvtColor(image.get_data(), cv2.COLOR_RGBA2RGB)
        point_cloud_np = point_cloud.get_data()

        video_writer.write(image_np)

        # Save depth data as .npy file (with frame number) within a folder
        np.save("DepthData/depth_{}.npy".format(frame_counter), point_cloud_np)

        # Show images
        cv2.namedWindow('ZED camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('ZED', image_np)
        cv2.waitKey(1)

        frame_counter += 1  # Increment frame counter

        key = cv2.waitKey(1)
        # Press 'q' or 'esc' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            break


# Close the video writer
video_writer.release()
# Close the video file
cv2.destroyAllWindows()
# Close the camera
zed.close()
