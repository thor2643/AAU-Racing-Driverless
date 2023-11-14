import cv2

def remove_frames_from_start(input_video, output_video, frames_to_remove_start, frames_to_remove_end):
    cap = cv2.VideoCapture(input_video)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the new .avi file
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if frame_number >= frames_to_remove_start and frame_number <= frames_to_remove_end:
            out.write(frame)

    out.release()
    cap.release()

input_video = "Data_AccelerationTrack/3/Color.avi"  # Replace with your input video file
output_video = "Data_AccelerationTrack/3/NewColor2.avi"  # Replace with your desired output video file
frames_to_remove_start = 390  # start at this fram
frames_to_remove_end = 1525  # End at this frame

remove_frames_from_start(input_video, output_video, frames_to_remove_start,frames_to_remove_end )

print("Done!")
