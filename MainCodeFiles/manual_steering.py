from getkey import getkey, keys
import serial

ser = serial.Serial("/dev/ttyUSB1", timeout=5)

def manual_steering():#serial):
    global ser
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
            speed += 1
            
        elif key == keys.DOWN:
            print("down")
            speed -= 1
            
        elif key == keys.LEFT:
            print("left")
            angle += 1

        elif key == keys.RIGHT:
            print("right")
            angle -= 1

        elif key == "s":
            print("Stopping")
            angle = 95
            speed = 90

        elif key == "q":
            print("Exiting")
            print(f"A{95}V{90}")
            #serial.write(f"A{95}V{90}")
            break

        #print(f"A{angle}V{speed}")
        response = ser.readline().decode("utf-8").rstrip()
        ser.write(f"A{angle}V{speed}".encode("utf-8"))

        
        print(response)


manual_steering()#ser)