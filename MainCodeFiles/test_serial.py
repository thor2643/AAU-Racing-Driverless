import serial
import random
import time

# Connect to the serial port
ser = serial.Serial('/dev/ttyUSB1', 115200)  # Adjust the port and baud rate (/dev/ttyUSB0)

"""
while True:
    #i random number to send to ardu
    i = random.randint(0, 100)
    ser.write(f"Hello from Python! no {i}".encode('utf-8'))
    data = ser.readline().decode('utf-8').rstrip()
    print(f'Received data: {data}')
    time.sleep(0.33)

# Don't forget to close the serial port when done
ser.close()
"""

ser.write(f"Hello from Python! no".encode('utf-8'))
time.sleep(0.33)
i=0
while True:
    data = ser.readline().decode('utf-8').rstrip()
    ser.write(f"A{95}V{90}".encode('utf-8'))
    
    print(f'Received data: {data}')
    #time.sleep(0.33)
    i+=1

ser.close()