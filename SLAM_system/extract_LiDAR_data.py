import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import serial
import time
import math


"""
class ExtractLidarData:
    def __init__(self) -> None:
        pass
"""

def setup_lidar(port='COM4', baudrate=115200):
    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2)  # Allow time for the connection to stabilize

    # Initialize the LiDAR sensor using SCIP2.0 protocol
    #ser.write(b'SCIP2.0\n')
    
    time.sleep(1)  # Allow time for initialization
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    return ser

def read_lidar_data(ser):
    ser.write(b'BM\n')  # Start measurement command

    # Wait for a moment to collect data
    time.sleep(2)
    
    data_line = ser.readline().decode() #.strip()
    print(data_line)

    # Stop measurement command
    ser.write(b'QT\n')
    ser.readline()  # Discard the first line

    # Read LiDAR data
    lidar_data = []
    for _ in range(681):  # URG-04LX has 681 data points
        data_line = ser.readline().decode() #.strip()
        print(data_line)
        distance = int(data_line.split(',')[1])
        lidar_data.append(distance)

    return lidar_data

def close_lidar_connection(ser):
    ser.close()

def read_lidar_data_2(ser):
    #ser.write(b'MS0384038401103\n')  # Start measurement command
    #ser.write(b'GS0000001001\n')
    #ser.write(b'QT\n')
    #ser.write(b'BM\n')
    #ser.write(b'SCIP2.0\n')
    #ser.write(b'PP\n')
    ser.write(b'VV\n')
    #ser.write(b'HS0\n')
    
    # Wait for a moment to collect data
    time.sleep(2)
    
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    data_line = ser.readline().decode() #.strip()
    print(data_line)
    
def hex_to_six_bit_binary(hex_values):
    binary_values = [
        bin(int(''.join(pair), 16))[2:].zfill(6)
        for pair in hex_values
    ]
    return ''.join(binary_values)

def binary_to_decimal(binary_value):
    return int(binary_value, 2)

def subtract_30_from_hex_values(list_of_hex_values):
    modified_hex_values = []
    for hex_value_pair in list_of_hex_values:
        modified_hex_values.append([
            [
                hex((int(char, 16) - 0x30) % 256)[2:].upper().zfill(2)
                for char in hex_pair
            ]
            for hex_pair in hex_value_pair
        ])
    return modified_hex_values

def convert_from_char_to_hexadecimal(data_string):
    hex_values = [
        [hex(ord(char))[2:].zfill(2) for char in pair]
        for pair in [data_string[i:i+2] for i in range(0, len(data_string), 2)]
    ]
    return hex_values

def translate_measurement(measurement_string):
    result = []
    
    hex_values = convert_from_char_to_hexadecimal(measurement_string)

    hex_value_pair = subtract_30_from_hex_values([hex_values])

    for i in range(0, len(hex_value_pair[0])):
        if hex_value_pair[0][i][0] == '0':
            result.append(hex_to_six_bit_binary(hex_value_pair[0][i][1]))
        else:
            result.append(hex_to_six_bit_binary(hex_value_pair[0][i]))
    
    concatenated_binary = ''.join(result)
    
    # Split the concatenated binary into 12-bit chunks
    chunks = [concatenated_binary[i:i+12] for i in range(0, len(concatenated_binary), 12)]

    # Convert each 12-bit chunk to decimal
    decimal_values = [binary_to_decimal(chunk) for chunk in chunks]

    return decimal_values

def get_angles_from_stepsize(start_step= 0, end_step= 681):
    angles = []
    for i in range(start_step, end_step+1):
        angles.append(i * 0.3515625 - 135)
    return angles

def receive_lidar_data(ser, command_string, lines_to_skip=6, lines_to_read=2, measurement_delay=2, timeout=20):
    # Start measurement
    ser.write(b'BM\n')
    time.sleep(2)
    # Measurement command
    command = f'{command_string}'.encode('utf-8')
    ser.write(command)
    time.sleep(measurement_delay)  # Wait for a moment to collect data

    # Skip specified number of lines
    for _ in range(lines_to_skip):
        ser.readline()

    # Read lines until the start string is encountered
    while True:
        line = ser.readline().decode()
        print(f'line: {line}')
        if command_string in line:
            break

    # Read lines and accumulate data until the end condition is met or timeout
    length_data = ""
    start_time = time.time()
    skip_count = 0
    while True:
        line = ser.readline().decode()
        print(f'line: {line}')

        # Check for lines to skip
        if skip_count < lines_to_read:
            skip_count += 1
            continue

        # Check for end condition
        if line.strip() == '':
            break

        # Remove the last character from each line
        length_data += line.rstrip('\n')[:-1]

        # Check for timeout
        if time.time() - start_time > timeout:
            print("Timeout occurred. Exiting loop.")
            break

     # Stop measurement
    ser.write(b'QT\n')

    # Remove the newline/linefeed and Sum character from length_data (SCIP2.0 format: length_data__Sum__\n)
    length_data = length_data[:-1]

    return length_data

def plot_polar_point_cloud(data):
    # Extract angles and lengths from the data
    angles, lengths = zip(*data)
    
    # Convert angles to degrees
    angles_degrees = np.array(angles)
    print(f'angles: {angles_degrees}')
    # Create a polar plot
    plt.polar(np.radians(angles_degrees), lengths, 'bo', label='Data Points')

    # Set the title of the plot
    plt.title('Polar Point Cloud')

    # Show the plot
    plt.show()

def plot_3d_polar_point_cloud(data):
    # Extract angles and lengths from the data
    angles, lengths = zip(*data)
    
    # Convert polar coordinates to Cartesian coordinates
    x = np.array(lengths) * np.cos(np.array(np.radians(angles)))
    print(f'x: {x}')
    y = np.array(lengths) * np.sin(np.array(np.radians(angles)))
    print(f'y: {y}')
    z = np.zeros_like(x)  # Assuming z-coordinate is 0, adjust this based on your data

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D point cloud
    ax.scatter(x, y, z, c='b', marker='o', label='Data Points')

    ax.scatter(0, 0, 0, c='r', marker='o', label='LiDAR Center')

    # Set labels for each axis
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Set the title of the plot
    ax.set_title('3D Point Cloud using Polar Coordinates')

    # Show the plot
    plt.show()

def main2():
    ser = setup_lidar()
    length_data = receive_lidar_data(ser, 'MS0044072501100\n')
    #print(length_data)

    # Get angles
    angles = get_angles_from_stepsize(44, 725)

    # Replace this with your actual implementation
    result = translate_measurement(length_data)    
    #print(result)

    combined_data = list(zip(angles, result))
    print(combined_data)

    plot_polar_point_cloud(combined_data)
    # Close the serial connection
    ser.close()

if __name__ == "__main__":
    main2()


"""
if __name__ == "__main__":
    lidar_serial = setup_lidar()

    try:
        lidar_data = read_lidar_data_2(lidar_serial)
        #get_length_data_from_lidar(lidar_serial)
        #print("LiDAR Data:", lidar_data)
    finally:
        close_lidar_connection(lidar_serial)
"""