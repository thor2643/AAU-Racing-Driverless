import numpy as np
import math
import socket
import matplotlib.pyplot as plt
import time
import keyboard



class Lidar:
    def __init__(self, address):
        self.sensorAddr = address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.sensorAddr)
        
    def query(self, queryText):
        self.sock.sendall((chr(2) + queryText + chr(3)).encode('ascii'))
        reply = ''
        while True:
            data = self.sock.recv(16)
            reply += data.decode()
            if chr(3).encode('ascii') in data:
                break
        return reply[1:-1] # Removing the STX/ETX control characters.
    
    def close(self):
        self.sock.close()
        print('Closed connection')
    
    def startMeasurement(self):
        return self.query('sMN LMCstartmeas')
    
    def run(self):
        return self.query('sMN Run')
    
    def singleScan(self):
        return self.query('sRN LMDscandata')

    def startContinuousScan(self):
        return self.query('sEN LMDscandata 1')

    def readContinuousScan(self, callback):
        while True:
            reply = ''
            while True:
                data = self.sock.recv(16)
                reply += data.decode()
                if chr(3).encode('ascii') in data:
                    break
            if not callback(reply[1:-1]): # Removing the STX/ETX control characters.
                return
        
    def stopContinuousScan(self):
        return self.query('sEN LMDscandata 0')

    def decodeData(self, data):
        # Split the output by spaces and store it in a list
        data_list = data.split()

        # Extract the measured data of the channel of the LIDAR sensor from the list
        # The measured data starts from the 26th element and ends at the 26th + number of data element
        # The number of data is the 25th element of the list
        # The measured data are hexadecimal values that represent the distance values in millimeters
        measured_data = data_list[26:26+int(data_list[25], 16)]

        # Convert the measured data from hexadecimal to decimal and store it in a numpy array
        distance_array = np.array([int(x, 16) for x in measured_data])

        # Print the distance array
        print(distance_array)

    def to_cartesian(distances, angles):
        x = list(map(lambda r, t: r * math.cos(math.radians(t)), distances, angles))
        y = list(map(lambda r, t: r * math.sin(math.radians(t)), distances, angles))
        return (x, y)
    
    def parse_data(self, data):
        # Split the output by spaces and store it in a list
        data_list = data.split()

        # Extract the measured data of the channel of the LIDAR sensor from the list
        # The measured data starts from the 26th element and ends at the 26th + number of data element
        # The number of data is the 25th element of the list
        # The measured data are hexadecimal values that represent the distance values in millimeters
        measured_data = data_list[26:26+int(data_list[25], 16)]

        # Convert the measured data from hexadecimal to decimal and store it in a numpy array
        distance_array = np.array([int(x, 16) for x in measured_data])

        # Get the start angle and the angle increment from the list
        # The start angle is the 23rd element of the list
        # The angle increment is the 24th element of the list
        # The angles are hexadecimal values that represent the angles in radians
        start_angle = int(data_list[23], 16) / 10000
        angle_increment = int(data_list[24], 16) / 10000

        # Create an array of angles for each ray using the start angle and the angle increment
        angle_array = np.array([start_angle + (i * angle_increment) for i in range(len(distance_array))])

        # Return the distance array and the angle array
        return distance_array, angle_array

    # Plot the 2D point cloud using the distance values and the angles
    def plot_point_cloud(self, distance_array, angle_array):
        # Convert the polar coordinates to cartesian coordinates
        x_array = np.array(distance_array) * np.cos(np.array(np.radians(angle_array)))
        y_array = np.array(distance_array) * np.sin(np.array(np.radians(angle_array)))

        plt.axline((0, 0), (0, 1), linewidth=1, color='green')
        plt.axline((0, 0), (1, 0), linewidth=1, color='purple')

        plt.plot(0, 0, marker="o", markersize=2, markeredgecolor="red", markerfacecolor="red")
        
        # Plot the x and y values as a scatter plot
        plt.scatter(x_array, y_array, s=1, c='b')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('2D Point Cloud from LIDAR Data')
        plt.show()

    def find_cones(self, distance_array, angle_array, threshold = 50):
        x_array = distance_array * np.cos(np.radians(angle_array))
        y_array = distance_array * np.sin(np.radians(angle_array))

        # Create a list to store cluster indices
        clusters = []

        # Iterate through points
        for i in range(len(x_array)):
            point = (x_array[i], y_array[i])

            # Check if the point is close to any existing cluster
            found_cluster = False
            for cluster in clusters:
                for existing_point in cluster:
                    distance = np.linalg.norm(np.array(point) - np.array(existing_point))
                    if distance < threshold:
                        cluster.append(point)
                        found_cluster = True
                        break

            # If the point is not close to any existing cluster, create a new cluster
            if not found_cluster:
                clusters.append([point])

        # Convert clusters to NumPy arrays for further processing if needed
        cluster_arrays = [np.array(cluster) for cluster in clusters]
        # print(cluster_arrays)

        dist_array = []
        final_array = []
        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                distance = math.sqrt((clusters[i][j][0] - 0)**2 + (clusters[i][j][1] - 0)**2)
                dist_array.append(distance)

            minimum_val = min(dist_array)
            for k in range(len(dist_array)):
                if dist_array[k] == minimum_val:
                    index = k

            if i < len(clusters):
                final_array.append(clusters[i])
            #else:
                #print(f"Index out of range: i={i}, index={index}, len(clusters)={len(clusters)}, len(clusters[i])={len(clusters[i]) if i < len(clusters) else 'N/A'}")

        return final_array

# main():
lidar_address = ('192.168.10.28', 2112) # IP address, TCP port
lidar = Lidar(lidar_address)

#fig, ax = plt.subplots()
fig_2, ax_2 = plt.subplots()
#ax.set_aspect('equal') # Set the aspect ratio to 1
ax_2.set_aspect('equal')
scan = True

try:
    while scan:
        if keyboard.is_pressed('q'):
            plt.close('all')
            break
        
        single_scan_data = lidar.singleScan()

        distance_array, angle_array = lidar.parse_data(single_scan_data)

        points = lidar.find_cones(distance_array, angle_array)
        
        for point in points:
            if len(point) == 1 and isinstance(point[0], tuple) and len(point[0]) == 2:
                tuple_point = point[0]
                ax_2.scatter(tuple_point[0], tuple_point[1], s=1)

                
           
        x_array = np.array(distance_array) * np.cos(np.array(np.radians(angle_array)))
        y_array = np.array(distance_array) * np.sin(np.array(np.radians(angle_array)))

        plt.xlim([-5000, 5000]) # Set the x axis limits
        plt.ylim([-5000, 5000]) # Set the y axis limits

        #plt.axline((0, 0), (0, 1), linewidth=1, color='green')
        #plt.axline((0, 0), (1, 0), linewidth=1, color='purple')

        #ax.plot(0, 0, marker="o", markersize=2, markeredgecolor="red", markerfacecolor="red")
        ax_2.plot(0, 0, marker="o", markersize=2, markeredgecolor="red", markerfacecolor="red")

        # Plot the x and y values as a scatter plot
        #ax.scatter(x_array, y_array, s=1, c='b')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('2D Point Cloud from LIDAR Data')

        plt.pause(0.03)
        #ax.clear()
        ax_2.clear()

finally:
    print("stopped")
    lidar.close()

"""
try:
    #lidar.startMeasurement()
    #lidar.run()

    single_scan_data = lidar.singleScan()
    print("Single Scan Data:", single_scan_data)
    distance_array, angle_array = parse_data(single_scan_data)
    plot_point_cloud(distance_array, angle_array)

finally:
    # Ensure to stop the continuous scan and close the connection
    lidar.close()
"""