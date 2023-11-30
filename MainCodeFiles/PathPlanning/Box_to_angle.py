import DelanayTriangles_NEW as DT

import numpy as np
import cv2

def list_to_nx3_array(list):
    return np.array(list).reshape(-1,3)


def boxes_to_cone_pos(bounding_box_list,point_cloud_xyz):
    #bounding_box_list=[[x1,y1,x2,y2],type] where types: 0=yellow, 1=blue, 2=orange, 3=large orange
    #point_cloud_xyz is a numpy array of shape (height,width,3) where the last dimension is the x,y,z coordinates of the point cloud
    #returns a list of the cones 2D position and type: [[x1,y1,type1],[x2,y2,type2],...] where x and y are in (mili/centi)meters

    cones_pos_type=[]

    #gets the cone's 2D position and type for evry bounding box in the list:
    for i, box in enumerate(bounding_box_list):
        #The pixel coordinates of the bounding box
            #the first point is the top left corner
        x_1=box[0][0]
        y_1=box[0][1]
            #the second point is the bottom right corner
        x_2=box[0][2]
        y_2=box[0][3]
            #the cone type
        type_cone=box[1]
        
        #The pixel coordinates of the new bounding box that is 1/3 of the original bounding box
            #the new box is for getting the position of the cone from the point cloud
            #the buttom 1/3 of the original is the most reliable for getting the position of the cone
        new_y_1=int((y_2-y_1)*2/3+y_1)
        new_x_1=int((x_2-x_1)*1/3+x_1)
        new_x_2=int(-1*(x_2-x_1)*1/3+x_2)


        #get all the 3D x, y, z coordinates of the point cloud in the new bounding box:
            #save the 2D coordinates of the point cloud in the new bounding box:
        x_cone_pos_arr=point_cloud_xyz[new_y_1:y_2,new_x_1:new_x_2,0]
        y_cone_pos_arr_y=point_cloud_xyz[new_y_1:y_2,new_x_1:new_x_2,1]
        y_cone_pos_arr_z=point_cloud_xyz[new_y_1:y_2,new_x_1:new_x_2,2]
        y_cone_pos_arr_w_inf=np.sqrt(y_cone_pos_arr_y**2+y_cone_pos_arr_z**2)

        #get the median of the x and y coordinates in 2D from the new bounding box, without the nan and inf values, 
            # and save it to the list of the cones 2D position and type:
        x_cone_pos=np.nanmedian(x_cone_pos_arr[np.isfinite(x_cone_pos_arr)])
        y_cone_pos=np.nanmedian(y_cone_pos_arr_w_inf[np.isfinite(y_cone_pos_arr_w_inf)])
        if np.isfinite(x_cone_pos) and np.isfinite(y_cone_pos):
            cones_pos_type.append([x_cone_pos,y_cone_pos,type_cone])
    return cones_pos_type

def boxes_to_midtpoints(bounding_box_list,point_cloud_xyz):
    #bounding_box_list=[[x1,y1,x2,y2],type] where types: 0=yellow, 1=blue, 2=orange, 3=large orange
    #point_cloud_xyz is a numpy array of shape (height,width,3) where the last dimension is the x,y,z coordinates of the point cloud
    #returns a list of the midpoints of the triangles that are made by two points of different color: [[x1,y1],[x2,y2],...] where x and y are in (mili/centi)meters

    #convert the bounding box list to a list of the cones 2D position and type: [[x1,y1,type1],[x2,y2,type2],...] where x and y are in (mili/centi)meters
    cone_coords_list=boxes_to_cone_pos(bounding_box_list,point_cloud_xyz)

    #Remove color information and make it to a numpy array of shape (n,2):
    bounding_box_arr_without_color=np.array(cone_coords_list).reshape(-1,3)[:,0:2]

    # Generate the Delaunay triangulation and filter the triangles that are made by three points of the same color
    tri=DT.delaunay_triangles_filtered(cone_coords_list, bounding_box_arr_without_color)


    # Find the midpoints of the triangles that are made by two points of different color
    midpoints = DT.find_midpoints(tri, cone_coords_list)
    return midpoints

def get_steering_angle(point_pos,car_length):
    #point_pos: position of the point the car shall be heading towards
    #returns: steering angle in degrees
    x_1=point_pos[0]
    y_1=point_pos[1]
    radius=(x_1**2+y_1**2)/(2*x_1)
    steering_angle=np.arctan(car_length/radius)
    return np.rad2deg(steering_angle)

def boxes_to_steering_angle(bounding_box_list,point_cloud_xyz,car_length,old_steering_angle=0,number_of_midpoints=4,weight_p0=3/5,weight_p1=1/5):
    #bounding_box_list=[[x1,y1,x2,y2],type] where types: 0=yellow, 1=blue, 2=orange, 3=large orange
    #point_cloud_xyz is a numpy array of shape (height,width,3) where the last dimension is the x,y,z coordinates of the point cloud
    #returns the steering angle in degrees

    #get the midpoints of the triangles that are made by two points of different color:
    midpoints=boxes_to_midtpoints(bounding_box_list,point_cloud_xyz)
    midpoints=np.array(midpoints).reshape(-1,2)

    #get the number_of_midpoints first midpoints:
    mid_points_next=midpoints[:number_of_midpoints,:]

    #get the weighted streing angle of the midpoints:
    angles_x=[]
    angles_list=[]
    sum_weights=0
    if len(mid_points_next)<number_of_midpoints:
        number_of_midpoints=len(mid_points_next)
    for i in range(number_of_midpoints):    
        angles_x.append(get_steering_angle(mid_points_next[i,:],car_length)*weight_p0)
        sum_weights+=weight_p0
        if weight_p0==3/5:
            weight_p0=weight_p1*2
        if i<(len(mid_points_next)-2):
            weight_p0=weight_p0/2
    steering_angle=np.sum(angles_x)

    #check if the steering angle is too big:
    if steering_angle>90 or steering_angle<-90:
        steering_angle=old_steering_angle
    elif steering_angle>25:
        steering_angle=25
    elif steering_angle<-25:
        steering_angle=-25
    
    #convert the steering angle to servo angle:
    servo_angle=-1.897*steering_angle+94.728

    return servo_angle,steering_angle


###TESTING:
#"""
import os

#parameters:
run_number = 1
frame_number = 150

#path to data files
data_path="C:\\Users\\3103e\\Documents\\GitHub\\AAU-Racing-Driverless\\ZED_camera\\Recordings_folder"

#Frame_150=[[[121, 419], [175, 490],0], [[403, 396], [423, 426],0], [[469, 388], [483, 410],0], [[507, 383], [519, 400],0], [[725, 388], [735, 404],1], [[752, 393], [764, 411],1], [[829, 402], [850, 429],1], [[1018, 430], [1059, 492],1]]
Frame_150=[[[121, 419,175, 490],0], [[403, 396,423, 426],0], [[469, 388,483, 410],0], [[507, 383,519, 400],0], [[725, 388,735, 404],1], [[752, 393,764, 411],1], [[829, 402,850, 429],1], [[1018, 430,1059, 492],1]]

#Frame_150=[[[121, 419], [175, 490],0], [[403, 396], [423, 426],0], [[829, 402], [850, 429],1], [[1018, 430], [1059, 492],1]]
#Frame_150=[[[121, 419], [175, 490],0], [[403, 396], [423, 426],0], [[1018, 430], [1059, 492],1]]
#Frame_150=[[[121, 419], [175, 490],0], [[829, 402], [850, 429],1]]

depth_data_path = os.path.join(data_path,"Run{}".format(run_number))
depth_arr=np.load(os.path.join(depth_data_path,"depth_{}.npy".format(frame_number)))

img_point_cloud_float=depth_arr[:,:,3]

# Convert the img_point_cloud_float array to a byte array
img_point_cloud_bytes = img_point_cloud_float.tobytes()

# Unpack the byte array into a 4D array of 4 bytes each
img_point_cloud_BGRA = np.frombuffer(img_point_cloud_bytes, dtype=np.uint8).reshape(img_point_cloud_float.shape + (4,))

point_cloud_frame=cv2.cvtColor(img_point_cloud_BGRA, cv2.COLOR_BGRA2RGBA)

servo_angle,steering_angle=boxes_to_steering_angle(Frame_150,depth_arr[:,:,0:3],700,0,4)
print(f"servo_angle={servo_angle}, steering_angle={steering_angle}")

points_cones=boxes_to_cone_pos(Frame_150,depth_arr[:,:,0:3])

midpoints=boxes_to_midtpoints(Frame_150,depth_arr[:,:,0:3])


#display the points with matplotlib:
import matplotlib.pyplot as plt
#plot
fig_1, ax_1 = plt.subplots()
ax_1.set_aspect('equal') # Set the aspect ratio to 1
for i in range(len(points_cones)):
    x, y, color = points_cones[i]
    if color==0:
        ax_1.scatter(x, y, c='gold', marker='o') # Create a scatter plot object
    elif color==1:
        ax_1.scatter(x, y, c='b', marker='o') # Create a scatter plot object

midpoints=np.array(midpoints).reshape(-1,2)
ax_1.scatter(midpoints[:,0],midpoints[:,1],c='r', marker='o')

cv2.imshow("Image_point_cloud", point_cloud_frame)
plt.show()
#"""
"""
####
#put this in the for loop in "def boxes_to_cone_pos()":
        #############
        if type_cone>1:
            #cv2.rectangle(point_cloud_frame, (x_pixel_min, y_pixel_max), (x_pixel_max, y_pixel_min), (255, 0, 0), 2)
            #cv2.rectangle(point_cloud_frame, (new_x_pixel_min, y_pixel_max), (new_x_pixel_max, new_y_pixel_min), (0, 255, 0), 2)
            cv2.rectangle(point_cloud_frame, (x_1, y_2), (x_2, y_1), (255, 0, 0), 2)
            cv2.rectangle(point_cloud_frame, (new_x_1, y_2), (new_x_2, new_y_1), (0, 255, 0), 2)
        else:
            cv2.rectangle(point_cloud_frame, (x_1, y_2), (x_2, y_1), (0, 255, 255), 2)
            cv2.rectangle(point_cloud_frame, (new_x_1, y_2), (new_x_2, new_y_1), (0, 255, 0), 2)
        ##############
"""
