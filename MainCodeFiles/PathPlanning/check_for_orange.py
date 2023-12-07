import numpy as np

def check_for_orange_cones(cones_pos_type, Track, number_of_orange_cones, ser, servo_angle, orange_seen_in_row, Wait_cone): #track = True (acceleration), False (Track drive)
    #print(f"cones_pos_type{cones_pos_type}")
    cones_pos_type=np.array(cones_pos_type).reshape(-1,3)
    #Removes all but the orange cones 
    orange_cones_pos_type=cones_pos_type[cones_pos_type[:,2]>1,:]

    #print(f"orange_cones_pos_type {orange_cones_pos_type}")
    #print(f"orange_cones_pos_type.shape[0] {orange_cones_pos_type.shape[0]}")

    if orange_cones_pos_type.shape[0] >= 2: #two or more cones must be seen at the same time
        orange_seen_in_row += 1
    else:
        orange_seen_in_row = 0

    #print(f"orange_seen_in_row {orange_seen_in_row}")
    if orange_seen_in_row == 3:
        number_of_orange_cones +=1
        Wait_cone = True

    if number_of_orange_cones > 9 and Track == False:
        ser.write(f"A{int(servo_angle)}V{90}".encode("utf-8"))
        print("Track finished")
        raise KeyboardInterrupt("Too many faulty path calculations")

    if number_of_orange_cones > 0 and Track == True:
        ser.write(f"A{int(servo_angle)}V{90}".encode("utf-8"))
        print("Track finished")
        raise KeyboardInterrupt("Too many faulty path calculations")

    return number_of_orange_cones, orange_seen_in_row, Wait_cone