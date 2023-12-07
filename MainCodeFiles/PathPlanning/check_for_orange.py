import numpy as np

def check_for_orange_cones(cones_pos_type, Track, number_of_orange_cones, ser, servo_angle, orange_seen_in_row, Wait_cone,STOP_for_orange, firstcone): #track = True (acceleration), False (Track drive)
    if STOP_for_orange == True:
        ser.write(f"A{int(servo_angle)}V{90}".encode("utf-8"))
        print("Track finished")
        raise KeyboardInterrupt("Too many faulty path calculations")

    #print(f"cones_pos_type{cones_pos_type}")
    cones_pos_type=np.array(cones_pos_type).reshape(-1,3)
    #Removes all but the orange cones 
    orange_cones_pos_type=cones_pos_type[cones_pos_type[:,2]>1,:]
    if orange_cones_pos_type.shape[0] >= 1 and firstcone == True:
        print(f"orange cone found, it is {orange_cones_pos_type[0][1]} m away.")
        firstcone = False

    if orange_cones_pos_type.shape[0] >= 2: #two or more cones must be seen at the same time
        orange_seen_in_row += 1
    else:
        orange_seen_in_row = 0

    #print(f"orange_seen_in_row {orange_seen_in_row}")
    if orange_seen_in_row == 3:
        number_of_orange_cones +=1
        print(number_of_orange_cones)
        orange_seen_in_row = 0
        if Track == False and number_of_orange_cones != 10:
            Wait_cone = True

    if (number_of_orange_cones > 0 and Track == True) or (number_of_orange_cones > 9 and Track == False):
        #if orange_cones_pos_type[0][1] <1.5:
        Wait_cone = True
        STOP_for_orange = True

    return number_of_orange_cones, orange_seen_in_row, Wait_cone, STOP_for_orange, firstcone
