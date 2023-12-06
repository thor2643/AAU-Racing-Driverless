from getkey import getkey, keys
import serial
import time

#ser = serial.Serial("/dev/ttyUSB0", baudrate=115200, dsrdtr=None)
#ser.setRTS(False)
#ser.setDTR(False)
#serial.Serial('COM4', 115200, dsrdtr=None)

def manual_steering(ser):
    print("You can now control the car using the arrow keys")
    #Maks højre:   45 
    #Ligeud:       95 (cirkus)
    #Maks venstre: 145
    #0 = fuld fart baglæns
    #90 = Stoppet
    #180 = fuld fart frem
    angle = 95
    speed = 90

    while True:
        key = getkey()
        if key == keys.UP:
            print("up")
            speed += 5
            
        elif key == keys.DOWN:
            print("down")
            speed -= 5
            
        elif key == keys.LEFT:
            print("left")
            angle += 2

        elif key == keys.RIGHT:
            print("right")
            angle -= 2

        elif key == "s":
            print("Emergency Stopping")
            ser.write("stop".encode("utf-8"))
            angle = 95
            speed = 90
            break

        elif key == "q":
            print("Exiting")
            print(f"A{95}V{90}")
            ser.write(f"A{95}V{90}".encode("utf-8"))
            angle = 95
            speed = 90
            break
            #time.sleep(5)
            #break

        #print(f"A{angle}V{speed}")
        ser.write(f"A{angle}V{speed}".encode("utf-8"))
        #response = ser.readline().decode("utf-8").rstrip()

        
        #print(response)


#manual_steering(ser)#ser)#ser)
#ser.close()