import serial
import time
import binascii

class ExtractLidarData:
    def __init__(self) -> None:
        pass

def setup_lidar(port='COM4', baudrate=115200):
    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2)  # Allow time for the connection to stabilize
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
    ser.write(b'MS0384038401101\n')  # Start measurement command
    #ser.write(b'GS0000001001\n')
    #ser.write(b'QT\n')
    #ser.write(b'BM\n')
    #ser.write(b'SCIP2.0\n')
    #ser.write(b'PP\n')
    #ser.write(b'HS0\n')
    """
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
    """
    
"""
def split_into_pairs(input_list):
    return [input_list[i:i + 2] for i in range(0, len(input_list), 2)]

def convert_from_char_to_hexadecimal(data_string):
    hex_value = binascii.hexlify(data_string.encode('utf-8'))
    hex_value = hex_value.decode('utf-8')  # Convert bytes to string
    return hex_value

def subtract_30_from_hex_values(list_of_hex_values):
    for i in range(len(list_of_hex_values)):
        list_of_hex_values[i][0] = hex(int(list_of_hex_values[i][0], 16) - 3)[2:] # remove '0x' from the start
        list_of_hex_values[i][1] = hex(int(list_of_hex_values[i][1], 16) - 0)[2:] # remove '0x' from the start
    return list_of_hex_values

def binary_to_decimal(binary_value):
    return int(binary_value, 2)

def hex_to_six_bit_binary(hex_value):
    binary_value = bin(int(hex_value, 16))[2:]
    return binary_value.zfill(6)  # Zero-fill to ensure it's 6 bits

def translate_measurement(measurement):
    meas_hex_value = convert_from_char_to_hexadecimal(measurement)
    hex_value_list = list(meas_hex_value)
    final_list = split_into_pairs(hex_value_list)
    final_list = subtract_30_from_hex_values(final_list)

    result = []
    for sublist in final_list:
        if sublist[0] == '0':
            result.append(hex_to_six_bit_binary(sublist[1]))
        else:
            result.append(hex_to_six_bit_binary(''.join(sublist)))

    concatenated_binary = ''.join(result)
    decimal_value = binary_to_decimal(concatenated_binary)
    display_measurement = f'{measurement} = {decimal_value}mm'

    return display_measurement

# Example usage
result = translate_measurement('m^')
print(result)
"""

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
    for i in range(start_step, end_step):
        angles.append(i * 0.3515625)
    return angles

def get_measurement_string(start_step= 0, end_step= 681):
    pass

def get_measurement(stepsize):
    angles = get_angles_from_stepsize()
    measurement = translate_measurement()

def receive_lidar_data():
    # Assume this is a function that reads lines from your data source
    # Replace this with your actual implementation
    data_source = get_data_source()  # You should replace get_data_source with your actual data source

    # Variables to keep track of the number of lines to skip and the accumulated data
    lines_to_skip = 5
    length_data = ""

    # Read lines to skip
    for _ in range(lines_to_skip):
        data_source.readline()

    # Read lines and accumulate data until there is no more
    while True:
        line = data_source.readline()

def main():
    length_data = ""  # Initialize an empty string to store length data
    lf_count = 0  # Count the consecutive linefeeds

    # Continue receiving length data until two consecutive linefeeds are encountered
    while lf_count < 2:
        new_data = receive_lidar_data()

        # Check if the new data is valid (you may need to adjust this condition)
        if new_data is not None:
            length_data += new_data

            # Check for linefeeds in the new data
            lf_count += new_data.count('\n')

    # Once two consecutive linefeeds are encountered, use your translate function on the accumulated length data
    result = translate_measurement(length_data)
    print(result)

if __name__ == "__main__":
    main()


"""
if __name__ == "__main__":
    lidar_serial = setup_lidar()

    try:
        lidar_data = read_lidar_data_2(lidar_serial)
        get_length_data_from_lidar(lidar_serial)
        print("LiDAR Data:", lidar_data)
    finally:
        close_lidar_connection(lidar_serial)
"""
