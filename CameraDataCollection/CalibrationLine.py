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

        self.codec = cv2.VideoWriter_fourcc(*'XVID')
        self.color_out = cv2.VideoWriter(self.color_output_file, self.codec, self.frame_rate, (self.width, self.height))

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

    # Create an align object for depth-color alignment
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
                
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            cv2.line(color_image, pt1=(int((color_image.shape[1]/2)),0), pt2=(int((color_image.shape[1]/2)),int((color_image.shape[0]))), color=(0,128,255), thickness=3)

            cv2.imshow('RealSense', color_image)
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
