import pyrealsense2 as rs
import os
import numpy as np
import cv2

class CameraConfig:
    def __init__(self, width=640, height=480, frame_rate=60, color_output_file=None, depth_output_file=None):
        self.width = width
        self.height = height
        self.frame_rate = frame_rate

        # Set custom output file names if provided, or use defaults
        self.color_output_file = color_output_file or "color_video.avi"
        #self.depth_output_file = depth_output_file or "depth_video.avi"

        self.codec = cv2.VideoWriter_fourcc(*'XVID')
        self.color_out = cv2.VideoWriter(self.color_output_file, self.codec, self.frame_rate, (self.width, self.height))
        #self.depth_out = cv2.VideoWriter(self.depth_output_file, self.codec, self.frame_rate, (self.width, self.height))


    def configure_pipeline(self):
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break

        if not found_rgb:
            print("The code requires a Depth camera with a Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.frame_rate)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.frame_rate)

        return pipeline, config

def main():
    camera_config = CameraConfig(width=640, height=480, frame_rate=30, color_output_file="Color.avi")

    # Create the "DepthData" directory if it doesn't exist
    if not os.path.exists("DepthData"):
        os.makedirs("DepthData")

    # Configure depth and color streams using the camera configuration class. The pipeline and config are now initialized as objects.
    pipeline, config = camera_config.configure_pipeline()
    pipeline.start(config)

    frame_counter = 0  # Initialize frame counter

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_INFERNO)

            # Check if the depth and color image can be stacked horizontally
            if depth_image.shape == color_image.shape:
                images = np.hstack((color_image, depth_colormap))
            else:
                #resize the color image to match the depth image
                color_image = cv2.resize(color_image, dsize=(depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((color_image, depth_colormap))

            # Save the images in video files (with frame number)
            camera_config.color_out.write(color_image)

            # Save depth data as .npy file (with frame number) within a folder
            np.save("DepthData/depth_{}.npy".format(frame_counter), depth_image)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

            frame_counter += 1  # Increment frame counter

            key = cv2.waitKey(1)
            # Press 'q' or 'esc' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        # Stop streaming
        pipeline.stop()
        # Release the video writer objects
        camera_config.color_out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()