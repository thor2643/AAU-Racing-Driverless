import serial
#print("hej")
ser = serial.Serial('/dev/ttyUSB0' ,115200)

while True:
    #ser.write("text")
    data = ser.readline().decode('utf-8').rstrip()
    print(f'Data: {data}')

ser.close()