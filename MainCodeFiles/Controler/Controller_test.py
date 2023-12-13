import serial
import time

#--------------------Initialize serial communication--------------------#
ser=serial.Serial("COM4",115200)
i=95
grow=1
while True:
    if i>145:
        grow=0
    elif i<95:
        grow=1
    
    if grow==1:
        i+=1
    else:
        i-=1
    data=f"A{i}V{i+5}"
    ser.write(data.encode("utf-8"))
    time.sleep(0.03)
    #recv_data=ser.readline().decode('utf-8').rstrip()
    #print(recv_data)
    #print(data)
