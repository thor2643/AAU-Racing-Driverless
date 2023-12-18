# Import the required libraries
#import Delanay_Triangles_copy as DT
import DelanayTriangles_copy_2 as DT
import numpy as np
import matplotlib.pyplot as plt

#perameters:
orientation_global=np.deg2rad(90)#rad
pos_point=np.array([0,0]) #x,y
dist_between_cones=3 #m

def stright_points(start_point,orientation,lenght,apppend_yellow,apppend_blue):
    global pos_point
    pos_point=[start_point[0]+np.cos(orientation)*lenght,start_point[1]+np.sin(orientation)*lenght]
    
    yellow_points_=np.array([[start_point[0],start_point[1]]])
    blue_points_=np.empty((0,2))

    dist=dist_between_cones
    while dist<=lenght:
        #Yellow points
        x_cor_y=yellow_points_[-1][0]+np.cos(orientation)*dist_between_cones
        y_cor_y=yellow_points_[-1][1]+np.sin(orientation)*dist_between_cones
        yellow_points_=np.append(yellow_points_,[[x_cor_y,y_cor_y]],axis=0)

        x_cor_b=yellow_points_[-1][0]+np.cos(orientation-np.deg2rad(90))*dist_between_cones
        y_cor_b=yellow_points_[-1][1]+np.sin(orientation-np.deg2rad(90))*dist_between_cones
        blue_points_=np.append(blue_points_,[[x_cor_b,y_cor_b]],axis=0)
        
        dist+=dist_between_cones
    apppend_yellow=np.append(apppend_yellow,yellow_points_[1:],axis=0)
    apppend_blue=np.append(apppend_blue,blue_points_,axis=0)
    return apppend_yellow,apppend_blue


def turn_points(start_point,orientation,dist_between_cones_in_turn,turn_radius_outer,turn_angle,apppend_yellow,apppend_blue,right_turn=True):
    global pos_point, orientation_global

    turn_radius_iner=turn_radius_outer-dist_between_cones

    #find the center point of the turn
    if right_turn==True:
        #center point 90 deg to the right of the orientation and the distance of the outer turn radius
        center_point_turn_circel=[start_point[0]+(np.cos(orientation+np.deg2rad(-90)))*turn_radius_outer,start_point[1]+(np.sin(orientation+np.deg2rad(-90)))*turn_radius_outer]
        #angle between the vector from the center to start point and the x-axis
        off_set_angle=orientation+np.deg2rad(90)

        #update the global position and orientation
        pos_point=[center_point_turn_circel[0]+np.cos(off_set_angle-turn_angle)*turn_radius_outer,center_point_turn_circel[1]+np.sin(off_set_angle-turn_angle)*turn_radius_outer]
        orientation_global-=turn_angle
    else:
        #center point 90 deg to the left of the orientation and the distance of the outer turn radius - dist_between_cones to get the iner turn radius
        center_point_turn_circel=[start_point[0]+(np.cos(orientation+np.deg2rad(90)))*turn_radius_iner,start_point[1]+(np.sin(orientation+np.deg2rad(90)))*turn_radius_iner]
        #angle between the vector from the center to start point and the x-axis
        off_set_angle=orientation+np.deg2rad(-90)

        #update the global position and orientation
        pos_point=[center_point_turn_circel[0]+np.cos(off_set_angle+turn_angle)*turn_radius_iner,center_point_turn_circel[1]+np.sin(off_set_angle+turn_angle)*turn_radius_iner]
        orientation_global+=turn_angle

    #the angle between each cone on the outer circle
    theta_outer=dist_between_cones_in_turn/(turn_radius_outer)
    #the angle between each cone on the iner circle
    theta_iner=dist_between_cones_in_turn/(turn_radius_iner)

    #empty array for the points on the outer circle
    yellow_points_=np.empty((0,2))
    blue_points_=np.empty((0,2))
    #generate points on the outer circle:
    angle_turned_outer=theta_outer
    angle_turned_iner=theta_iner
    #angle_turned_outer=0
    #angle_turned_iner=0
    while angle_turned_outer<=turn_angle:
        if right_turn==True:
            point_on_outer_circel=center_point_turn_circel+np.array([np.cos(off_set_angle-angle_turned_outer),np.sin(off_set_angle-angle_turned_outer)])*turn_radius_outer
            yellow_points_=np.append(yellow_points_,[point_on_outer_circel],axis=0)
            angle_turned_outer+=theta_outer
        else:
            point_on_outer_circel=center_point_turn_circel+np.array([np.cos(off_set_angle+angle_turned_outer),np.sin(off_set_angle+angle_turned_outer)])*turn_radius_outer
            blue_points_=np.append(blue_points_,[point_on_outer_circel],axis=0)
            angle_turned_outer+=theta_outer
    
    while angle_turned_iner<=turn_angle:
        if right_turn==True:
            point_on_iner_circel=center_point_turn_circel+np.array([np.cos(off_set_angle-angle_turned_iner),np.sin(off_set_angle-angle_turned_iner)])*turn_radius_iner
            blue_points_=np.append(blue_points_,[point_on_iner_circel],axis=0)
            angle_turned_iner+=theta_iner
        else:
            point_on_iner_circel=center_point_turn_circel+np.array([np.cos(off_set_angle+angle_turned_iner),np.sin(off_set_angle+angle_turned_iner)])*turn_radius_iner
            yellow_points_=np.append(yellow_points_,[point_on_iner_circel],axis=0)
            angle_turned_iner+=theta_iner

    apppend_yellow=np.append(apppend_yellow,yellow_points_,axis=0)
    apppend_blue=np.append(apppend_blue,blue_points_,axis=0)
    return apppend_yellow,apppend_blue

yellow_points=np.empty((0,2))
blue_points=np.empty((0,2))

def race_track_0():
    #make a arrays for the yellow and blue points
    yellow_points=np.empty((0,2))
    blue_points=np.empty((0,2))

    #sthight - section 1: 27 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,27,yellow_points,blue_points)
    
    
    return yellow_points,blue_points


def race_track_1():
    #make a arrays for the yellow and blue points
    yellow_points=np.empty((0,2))
    blue_points=np.empty((0,2))

    #sthight - section 1: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,12,yellow_points,blue_points)

    #turn - section 2: 90 deg right 12 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,3,12,np.deg2rad(90),yellow_points,blue_points,True)

    #sthight - section 3: 4 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,4,yellow_points,blue_points)

    #turn - section 5: 160 deg left 9 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,3,9,np.deg2rad(160),yellow_points,blue_points,False)

    #sthight - section 6: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,12,yellow_points,blue_points)

    #turn - section 7: 120 deg right 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,3,6,np.deg2rad(120),yellow_points,blue_points,True)

    #sthight - section 8: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,12,yellow_points,blue_points)

    #turn - section 9: 45 deg right 9 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,3,9,np.deg2rad(45),yellow_points,blue_points,True)

    #sthight - section 10: 14 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,14,yellow_points,blue_points)

    #turn - section 11: 120 deg right 12 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,3,12,np.deg2rad(120),yellow_points,blue_points,True)

    #sthight - section 12: 8 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,8,yellow_points,blue_points)

    #turn - section 13: 90 deg left 12 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,3,9,np.deg2rad(90),yellow_points,blue_points,False)

    #turn - section 14: 90 deg right 12 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,3,12,np.deg2rad(90),yellow_points,blue_points,True)

    #sthight - section 15: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,12,yellow_points,blue_points)

    #turn - section 16: 90 deg right 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,3,6,np.deg2rad(90),yellow_points,blue_points,True)

    #sthight - section 17: 8 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,8,yellow_points,blue_points)

    #turn - section 18: 120 deg left 9 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,3,9,np.deg2rad(120),yellow_points,blue_points,False)

    #turn - section 19: 90 deg right 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,3,6,np.deg2rad(175),yellow_points,blue_points,True)


    return yellow_points,blue_points

def race_track_2():
    #make a arrays for the yellow and blue points
    yellow_points=np.empty((0,2))
    blue_points=np.empty((0,2))

    #sthight - section 1: 20 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,9,yellow_points,blue_points)

    #turn - section 2: 90 deg right 8 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(90),yellow_points,blue_points,True)

    #Turn - section 3: 45 deg left 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(45),yellow_points,blue_points,False)

    #sthight - section 4: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,12,yellow_points,blue_points)

    #turn - section 5: 135 deg left 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(135),yellow_points,blue_points,False)

    #sthight - section 6: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,18,yellow_points,blue_points)

    #turn - section 7: 135 deg right 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(135),yellow_points,blue_points,True)

    #turn - section 8: 180 deg left 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(180),yellow_points,blue_points,False)

    #sthight - section 9: 24 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,33,yellow_points,blue_points)
    
    #turn - section 10: 90 deg left 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(90),yellow_points,blue_points,False)

    #turn - section 11: 90 deg right 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(90),yellow_points,blue_points,True)

    #sthight - section 12: 24 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,18,yellow_points,blue_points)

    #turn - section 13: 135 deg left 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(135),yellow_points,blue_points,False)

    #sthight - section 14: 21 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,21,yellow_points,blue_points)

    #turn - section 15: 135 deg right 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(90),yellow_points,blue_points,True)

    #turn - section 16: 180 deg left 12 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,12,np.deg2rad(180),yellow_points,blue_points,False)

    #sthight - section 17: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,12,yellow_points,blue_points)

    yellow_points=np.append(yellow_points,[[16,35.4]],axis=0)
    yellow_points=np.append(yellow_points,[[-14.5,55.6]],axis=0)
    yellow_points=np.append(yellow_points,[[-37.5,25.6]],axis=0)
    yellow_points=np.append(yellow_points,[[-46.2,-6.8]],axis=0)
    yellow_points=np.append(yellow_points,[[12,19]],axis=0)

    blue_points=np.append(blue_points,[[-21,-15]],axis=0)
    blue_points=np.append(blue_points,[[-17,57.5]],axis=0)
    blue_points=np.append(blue_points,[[3.3,-14]],axis=0)
    
    return yellow_points,blue_points    

def race_track_3():
    #make a arrays for the yellow and blue points
    yellow_points=np.empty((0,2))
    blue_points=np.empty((0,2))

    #turn - section 1: 90 deg right 12 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,12,np.deg2rad(90),yellow_points,blue_points,True)

    #turn - section 2: 180 deg left 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(180),yellow_points,blue_points,False)

    #turn - section 3: 90 deg right 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(90),yellow_points,blue_points,True)

    #sthight - section 4: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,12,yellow_points,blue_points)

    #turn - section 5: 135 deg right 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(135),yellow_points,blue_points,True)

    #sthight - section 6: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,12,yellow_points,blue_points)

    #turn - section 7: 135 deg left 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(135),yellow_points,blue_points,False)

    #turn - section 8: 180 deg right 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(180),yellow_points,blue_points,True)

    #sthight - section 9: 18 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,18,yellow_points,blue_points)

    #turn - section 10: 45 deg left 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(45),yellow_points,blue_points,False)

    #sthight - section 11: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,12,yellow_points,blue_points)

    #turn - section 12: 135 deg right 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(180),yellow_points,blue_points,True)

    #sthight - section 13: 6 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,6,yellow_points,blue_points)

    #turn - section 14: 135 deg left 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(90),yellow_points,blue_points,False)

    #sthight - section 15: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,24,yellow_points,blue_points)

    #turn - section 16: 135 deg right 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(45),yellow_points,blue_points,True)

    #sthight - section 17: 12 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,9,yellow_points,blue_points)

    #turn - section 18: 90 deg right 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(90),yellow_points,blue_points,True)

    #sthight - section 19: 6 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,3,yellow_points,blue_points)

    yellow_points=np.append(yellow_points,[[11.8,22]],axis=0)
    yellow_points=np.append(yellow_points,[[6,28.5]],axis=0)
    yellow_points=np.append(yellow_points,[[20.2,45]],axis=0)
    yellow_points=np.append(yellow_points,[[36.7,40.7]],axis=0)
    yellow_points=np.append(yellow_points,[[52.7,40]],axis=0)
    yellow_points=np.append(yellow_points,[[50,1.5]],axis=0)
    yellow_points=np.append(yellow_points,[[39.5,5]],axis=0)


    blue_points=np.append(blue_points,[[18,43]],axis=0)
    blue_points=np.append(blue_points,[[49.7,40]],axis=0)
    blue_points=np.append(blue_points,[[52.3,17.8]],axis=0)
    blue_points=np.append(blue_points,[[52.7,3]],axis=0)
    blue_points=np.append(blue_points,[[16.8,-11.3]],axis=0)
    blue_points=np.append(blue_points,[[2.7,-6]],axis=0)

    return yellow_points,blue_points

def race_track_5():
    #make a arrays for the yellow and blue points
    yellow_points=np.empty((0,2))
    blue_points=np.empty((0,2))

    #sthight - section 1: 20 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,33,yellow_points,blue_points)

    #turn - section 2: 90 deg left 12 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,12,np.deg2rad(90),yellow_points,blue_points,False)
    
    #sthight - section 1: 15 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,24,yellow_points,blue_points)
    
    #turn - section 2: 90 deg left 6 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(90),yellow_points,blue_points,False)
    
    #sthight - section 1: 15 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,18,yellow_points,blue_points)
    
    #turn - section 2: 180 deg left 6 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(180),yellow_points,blue_points,False)

    #sthight - section 1: 2 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,6,yellow_points,blue_points)
    
    #turn - section 2: 180 deg right 8 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(180),yellow_points,blue_points,True)
    
    #sthight - section 1: 20 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,20,yellow_points,blue_points)
    
    #turn - section 2: 90 deg right 10 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,10,np.deg2rad(90),yellow_points,blue_points,True)
    
    #sthight - section 1: 15 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,7,yellow_points,blue_points)
    
    #turn - section 2: 120 deg left 8 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(130),yellow_points,blue_points,False)

    #sthight - section 1: 6 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,10,yellow_points,blue_points)
    
    #turn - section 2: 120 deg left 8 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(130),yellow_points,blue_points,False)
    
    #turn - section 2: 150 deg right 8 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(150),yellow_points,blue_points,True)#
    
    #turn - section 2: 215 deg left 8 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(215),yellow_points,blue_points,False)
    
    #turn - section 2: 60 deg right 8 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(60),yellow_points,blue_points,True)

    
    blue_points=np.append(blue_points,[[-39,38.3]],axis=0)
    blue_points=np.append(blue_points,[[-17,26.5]],axis=0)
    #yellow_points=np.append(yellow_points,[[-30,22]],axis=0)
    yellow_points=np.append(yellow_points,[[-24.5,-3]],axis=0)
    yellow_points=np.append(yellow_points,[[-14,26]],axis=0)
    
    return yellow_points,blue_points

def race_track_4():
    #make a arrays for the yellow and blue points
    yellow_points=np.empty((0,2))
    blue_points=np.empty((0,2))

    #sthight - section 1: 21 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,21,yellow_points,blue_points)

    #turn - section 2: 90 deg right 12 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,12,np.deg2rad(90),yellow_points,blue_points,True)

    #sthight - section 3: 9 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,9,yellow_points,blue_points)

    #turn - section 4: 135 deg left 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(135),yellow_points,blue_points,False)

    #turn - section 5: 180 deg right 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(180),yellow_points,blue_points,True)

    #sthight - section 6: 21 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,21,yellow_points,blue_points)

    #turn - section 7: 90 deg right 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(90),yellow_points,blue_points,True)

    #sthight - section 8: 9 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,12,yellow_points,blue_points)

    #turn - section 9: 90 deg right 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(90),yellow_points,blue_points,True)

    #turn - section 10: 180 deg left 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(180),yellow_points,blue_points,False)

    #turn - section 11: 135 deg right 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(135),yellow_points,blue_points,True)

    #turn - section 12: 180 deg left 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(180),yellow_points,blue_points,False)

    #sthight - section 13: 21 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,21,yellow_points,blue_points)

    #turn - section 14: 135 deg right 8 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,8,np.deg2rad(135),yellow_points,blue_points,True)

    #sthight - section 15: 9 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,9,yellow_points,blue_points)

    #turn - section 16: 90 deg right 12 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,12,np.deg2rad(90),yellow_points,blue_points,True)

    #turn - section 17: 45 deg left 12 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,12,np.deg2rad(45),yellow_points,blue_points,False)

    #sthight - section 18: 9 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,15,yellow_points,blue_points)

    #turn - section 19: 90 deg eigth 7 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,7,np.deg2rad(90),yellow_points,blue_points,True)

    #sthight - section 20: 6 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,6,yellow_points,blue_points)

    yellow_points=np.append(yellow_points,[[13,33]],axis=0)
    yellow_points=np.append(yellow_points,[[26.2,0.5]],axis=0)

    blue_points=np.append(blue_points,[[49.5,-11.5]],axis=0)
    blue_points=np.append(blue_points,[[21.5,-13.8]],axis=0)
    blue_points=np.append(blue_points,[[26.2,-2.5]],axis=0)
    return yellow_points,blue_points


def race_track_6():
    #make a arrays for the yellow and blue points
    yellow_points=np.empty((0,2))
    blue_points=np.empty((0,2))
    
    #sthight - section 1: 3 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,3,yellow_points,blue_points)
    
    #turn - section 2: 180 deg Right 6 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(180),yellow_points,blue_points,True)
    
    #sthight - section 3: 3 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,6,yellow_points,blue_points)
    
    #turn - section 4: 45 deg left 6 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(45),yellow_points,blue_points,False)
    
    #turn - section 5: 90 deg Right 6 m radius 
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(90),yellow_points,blue_points,True)
    
    #sthight - section 6: 6 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,6,yellow_points,blue_points)
    
    #turn - section 7: 90 deg Right 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(90),yellow_points,blue_points,True)
    
    #sthight - section 8: 3 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,3,yellow_points,blue_points)

    #turn - section 9: 90 deg Right 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(95),yellow_points,blue_points,True)

    #turn - section 10: 45 deg left 6 m radius
    yellow_points,blue_points=turn_points(pos_point,orientation_global,2,6,np.deg2rad(45),yellow_points,blue_points,False)

    #sthight - section 11: 3 m
    yellow_points,blue_points=stright_points(pos_point,orientation_global,3,yellow_points,blue_points)
    
    yellow_points=np.append(yellow_points,[[12.5,-14.0]],axis=0)
    #37 gul
    #31 blå


    return yellow_points,blue_points

yellow_points,blue_points=race_track_6()

print(f"yellow_points=\n{yellow_points.tolist()}")
print(f"blue_points=\n{blue_points.tolist()}")
#make a list of all points
points=[]

for i in range(yellow_points.shape[0]):
    points.append([yellow_points[i,0],yellow_points[i,1],'yellow'])

for i in range(blue_points.shape[0]):
    points.append([blue_points[i,0],blue_points[i,1],'blue'])



def plot_points_with_delanay(point_array):
    # convert the point array to a NumPy array, without color information
    point_array_without_color = DT.remove_Colors(point_array)
    #print(point_array_without_color)


    # Use the filter function to remove the triangles that are made by three points of the same color
    tri = DT.delaunay_triangles_filtered(point_array, point_array_without_color,use_scipy=True)
    #print(tri)

    # Find the midpoints of the triangles that are made by two points of different color
    midpoints = DT.find_midpoints(tri, point_array)
    print(f"midpoints=\n{midpoints}")

    # Plot the points
    DT.plot_points(point_array, point_array_without_color, tri, midpoints)

#orange_cones = [[0,1.5],[3,1.5]]
orange_cones = [[0,0],[3,0]]
print(f"orange_cones=\n{orange_cones}")

def plot_kegler(yellow_points,blue_points, x_axis_length=3.5, y_axis_length=30):
    # Plot the yellow and the blue points in the same plot
    plt.plot(yellow_points[:, 0], yellow_points[:, 1], 'o', color='gold', label='Yellow Points')
    plt.plot(blue_points[:, 0], blue_points[:, 1], 'o', color='blue', label='Blue Points')
    
    #plot the orange cones
    plt.plot(np.array(orange_cones)[:, 0], np.array(orange_cones)[:, 1], 'o', color='darkorange', label='Orange Points')

    #Set equal scaling for both axes
    plt.axis('equal')
    
    # Set axis limits to control the length of each axis
    #plt.xlim(-0.5, x_axis_length)
    #plt.ylim(-1, y_axis_length+1)

    # Add labels, title, legend, etc., if needed
    plt.xlabel('X-axis [m]')
    plt.ylabel('Y-axis [m]')
    plt.title('Setup of the Trackdrive track')
    plt.legend()

    # Show the plot
    plt.show()


print(f"number of yellow cones={len(yellow_points)}")
print(f"number of blue cones={len(blue_points)}")
    
    
#plot_points_with_delanay(points)
plot_kegler(yellow_points,blue_points)