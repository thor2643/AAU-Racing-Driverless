import serial
import random

# Connect to the serial port
ser = serial.Serial('COM5', 115200)  # Adjust the port and baud rate (/dev/ttyUSB0)

while True:
    #i random number to send to ardu
    i = random.randint(0, 100)
    ser.write(f"Hello from Python! no {i}".encode('utf-8'))
    data = ser.readline().decode('utf-8').rstrip()
    print(f'Received data: {data}')

# Don't forget to close the serial port when done
ser.close()