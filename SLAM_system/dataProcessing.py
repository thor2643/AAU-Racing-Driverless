import numpy as np
import matplotlib.pyplot as plt

# Generate random IMU-like data
imu_timestamps = np.arange(0, 10, 0.1)  # 100 timestamps from 0 to 10 seconds
acceleration = np.random.randn(len(imu_timestamps), 3)  # Random acceleration data
gyroscope = np.random.randn(len(imu_timestamps), 3)  # Random gyroscope data

# Generate random 2D LiDAR-like data
lidar_timestamps = np.arange(0, 10, 1)  # LiDAR data every 1 second
lidar_angles = np.linspace(0, 2 * np.pi, 360)  # 2D LiDAR angles
lidar_distances = np.random.rand(360)  # Random distances

# Create lists to store the trajectory
trajectory = []

# Example: Synchronize IMU and LiDAR data
synchronized_data = []
for imu_time in imu_timestamps:
    # Find the closest LiDAR timestamp
    closest_lidar_time = lidar_timestamps[np.argmin(np.abs(lidar_timestamps - imu_time))]

    # Extract the corresponding IMU and LiDAR data
    imu_entry = [imu_time] + list(acceleration[int(imu_time * 10)] + gyroscope[int(imu_time * 10)])
    lidar_entry = [closest_lidar_time] + list(lidar_angles) + list(lidar_distances)

    synchronized_data.append((imu_entry, lidar_entry))

    # For visualization purposes, add a simplified trajectory estimation
    # In practice, replace this with your SLAM algorithm's pose estimation.
    trajectory.append((imu_time, imu_time * 0.1))  # (timestamp, x)

# Extract x, y coordinates from the LiDAR data
x = lidar_distances * np.cos(lidar_angles)
y = lidar_distances * np.sin(lidar_angles)

# Plot the LiDAR point cloud as a scatter plot with the trajectory overlay
plt.figure()
plt.scatter(x, y, c='b', s=1, label='LiDAR Data')
trajectory = np.array(trajectory)
plt.plot(trajectory[:, 1], trajectory[:, 1], 'r-', label='Robot Trajectory')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('LiDAR Point Cloud and Robot Trajectory')
plt.legend()
plt.grid(True)

plt.show()
