import DelanayTriangles_NEW as DT

import numpy as np
import cv2

def list_to_nx3_array(list):
    return np.array(list).reshape(-1,3)


def boxes_to_cone_pos(bounding_box_list,point_cloud_xyz):
    #bounding_box_list=[[x1,y1],[x2,y2],type] where types: 0=yellow, 1=blue, 2=orange, 3=large orange
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
        x_2=box[1][0]
        y_2=box[1][1]
            #the cone type
        type_cone=box[2]
        
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
    #bounding_box_list=[[x1,y1],[x2,y2],type] where types: 0=yellow, 1=blue, 2=orange, 3=large orange
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

def get_stering_angle(point_pos,car_length):
    #point_pos: position of the point the car shall be heading towards
    #returns: stering angle in degrees
    x_1=point_pos[0]
    y_1=point_pos[1]
    radius=(x_1**2+y_1**2)/(2*x_1)
    stering_angle=np.arctan(car_length/radius)
    return np.rad2deg(stering_angle)

def boxes_to_sterring_angle(bounding_box_list,point_cloud_xyz,car_length,old_steering_angle):
    #bounding_box_list=[[x1,y1],[x2,y2],type] where types: 0=yellow, 1=blue, 2=orange, 3=large orange
    #point_cloud_xyz is a numpy array of shape (height,width,3) where the last dimension is the x,y,z coordinates of the point cloud
    #returns the steering angle in degrees

    midpoints=boxes_to_midtpoints(bounding_box_list,point_cloud_xyz)

    

    return

#################
#old stuff:
#################
def plot_points_with_delanay(point_array):
    print(point_array)

    # convert the point array to a NumPy array, without color information
    point_array_without_color = DT.remove_Colors(point_array)
    print(point_array_without_color)

    point_array_without_color_new=np.array(point_array).reshape(-1,3)[:,:2].astype(np.float64)

    print(point_array_without_color_new)
    #convert from string to float
    point_array_without_color_new=point_array_without_color_new
    # Use the filter function to remove the triangles that are made by three points of the same color
    tri = DT.delaunay_triangles_filtered(point_array, point_array_without_color)
    #print(tri)

    # Find the midpoints of the triangles that are made by two points of different color
    midpoints = DT.find_midpoints(tri, point_array)

    # Plot the points
    DT.plot_points(point_array, point_array_without_color, tri, midpoints)

    return midpoints

import Delanay_Triangles as DT

import numpy as np
import cv2
import os

#parameters:
#run_number = 1
#frame_number = 150

#path to data files
data_path="C:\\Users\\3103e\\Documents\\GitHub\\AAU-Racing-Driverless\\ZED_camera\\Recordings_folder"

#video_path = os.path.join(data_path,"ZED_color_video_Run_{}.avi".format(run_number))



def display_frame_from_point_cloud_with_boxes_and_return_cone_pos(depth_arr_from_point_cloud,coords):
    depth_arr_xyz=depth_arr_from_point_cloud[:,:,0:3]
    img_point_cloud_float=depth_arr_from_point_cloud[:,:,3]

    # Convert the img_point_cloud_float array to a byte array
    img_point_cloud_bytes = img_point_cloud_float.tobytes()

    # Unpack the byte array into a 4D array of 4 bytes each
    img_point_cloud_BGRA = np.frombuffer(img_point_cloud_bytes, dtype=np.uint8).reshape(img_point_cloud_float.shape + (4,))

    point_cloud_frame=cv2.cvtColor(img_point_cloud_BGRA, cv2.COLOR_BGRA2RGBA)

    cones_pos_type=[]

    for i in range(len(coords)):
        font = cv2.FONT_HERSHEY_SIMPLEX

        x_pixel_max=max([coords[i][0][0],coords[i][1][0]])
        x_pixel_min=min([coords[i][0][0],coords[i][1][0]])
        y_pixel_max=max([coords[i][0][1],coords[i][1][1]])
        y_pixel_min=min([coords[i][0][1],coords[i][1][1]])

        new_y_pixel_min=int((y_pixel_max-y_pixel_min)*2/3+y_pixel_min)
        new_x_pixel_min=int((x_pixel_max-x_pixel_min)*1/3+x_pixel_min)
        new_x_pixel_max=int(-1*(x_pixel_max-x_pixel_min)*1/3+x_pixel_max)

        if x_pixel_max<point_cloud_frame.shape[1]/2:
            cv2.rectangle(point_cloud_frame, (x_pixel_min, y_pixel_max), (x_pixel_max, y_pixel_min), (255, 0, 0), 2)
            cv2.rectangle(point_cloud_frame, (new_x_pixel_min, y_pixel_max), (new_x_pixel_max, new_y_pixel_min), (0, 255, 0), 2)
            type_cone="blue"
        else:
            cv2.rectangle(point_cloud_frame, (x_pixel_min, y_pixel_max), (x_pixel_max, y_pixel_min), (0, 255, 255), 2)
            cv2.rectangle(point_cloud_frame, (new_x_pixel_min, y_pixel_max), (new_x_pixel_max, new_y_pixel_min), (0, 255, 0), 2)
            type_cone="yellow"

        x_cone_pos_arr=depth_arr_xyz[new_y_pixel_min:y_pixel_max,new_x_pixel_min:new_x_pixel_max,0]
        y_cone_pos_arr_y=depth_arr_xyz[new_y_pixel_min:y_pixel_max,new_x_pixel_min:new_x_pixel_max,1]
        y_cone_pos_arr_z=depth_arr_xyz[new_y_pixel_min:y_pixel_max,new_x_pixel_min:new_x_pixel_max,2]
        y_cone_pos_arr_w_inf=np.sqrt(y_cone_pos_arr_y**2+y_cone_pos_arr_z**2)

        x_cone_pos=np.nanmedian(x_cone_pos_arr[np.isfinite(x_cone_pos_arr)])
        y_cone_pos=np.nanmedian(y_cone_pos_arr_w_inf[np.isfinite(y_cone_pos_arr_w_inf)])

        #print(f"x_cone_pos={x_cone_pos}, y_cone_pos={y_cone_pos}")
        cv2.putText(point_cloud_frame, f"{i}", (x_pixel_max, y_pixel_min), font, 0.5, (0, 255, 0), 2)
        if np.isfinite(x_cone_pos) and np.isfinite(y_cone_pos):
            cones_pos_type.append([x_cone_pos,y_cone_pos,type_cone])
    
    cv2.imshow("Image_point_cloud", point_cloud_frame)
    return cones_pos_type


Frame_150=[150,[[109, 489], [172, 421]], [[1062, 491], [1019, 430]], [[850, 431], [830, 402]], [[426, 426], [403, 394]], [[469, 407], [483, 387]], [[751, 412], [763, 393]], [[734, 402], [724, 387]], [[520, 399], [507, 383]]]
Frame_200=[200,[[305, 403], [272, 449]], [[986, 459], [1017, 416]], [[827, 418], [840, 395]], [[446, 414], [464, 390]], [[514, 401], [526, 383]], [[784, 405], [794, 388]], [[767, 388], [760, 398]], [[557, 394], [568, 380]]]
Frame_250=[250,[[434, 439], [459, 401]], [[1066, 406], [1039, 444]], [[936, 391], [923, 416]], [[881, 390], [870, 402]], [[842, 383], [831, 396]], [[575, 390], [559, 411]], [[629, 383], [617, 399]], [[653, 381], [645, 394]]]
#Frame_250B=[250,[[654, 381], [646, 394]], [[629, 384], [618, 399]], [[574, 388], [559, 412]], [[458, 400], [435, 437]], [[1039, 446], [1062, 407]], [[937, 391], [921, 416]], [[882, 390], [871, 403]], [[841, 382], [832, 397]], [[826, 383], [819, 392]]]
Frame_300=[300,[[1265, 446], [1202, 532]], [[874, 404], [854, 434]], [[773, 395], [763, 412]], [[717, 389], [709, 401]], [[489, 384], [479, 399]], [[445, 386], [431, 409]], [[326, 396], [305, 432]]]
Frame_350=[350,[[47, 448], [1, 519]], [[1138, 425], [1095, 497]], [[872, 400], [857, 429]], [[783, 395], [771, 414]], [[400, 410], [382, 442]], [[493, 400], [479, 419]], [[525, 395], [516, 409]], [[756, 393], [746, 406]]]
Frame_400=[400,[[1176, 437], [1142, 492]], [[274, 478], [315, 426]], [[493, 434], [514, 406]], [[552, 421], [570, 401]], [[582, 409], [591, 398]], [[921, 411], [903, 441]], [[860, 404], [848, 423]], [[824, 400], [813, 414]]]
Frame_500=[500,[[22, 464], [0, 566]], [[469, 410], [488, 448]], [[564, 399], [580, 424]], [[608, 394], [620, 415]], [[640, 392], [648, 406]], [[1122, 424], [1152, 469]], [[964, 407], [978, 431]], [[900, 400], [912, 419]], [[868, 397], [877, 412]]]
Frame_600=[600,[[382, 495], [340, 426]], [[531, 400], [554, 434]], [[602, 391], [617, 411]], [[644, 386], [656, 404]], [[1122, 403], [1145, 439]], [[977, 392], [992, 416]], [[910, 387], [921, 406]], [[875, 386], [883, 400]]]
Frame_700=[700,[[729, 468], [755, 421]], [[597, 402], [614, 431]], [[550, 397], [563, 417]], [[521, 396], [531, 410]], [[482, 392], [492, 407]], [[457, 391], [470, 407]], [[443, 392], [452, 408]], [[410, 390], [420, 406]], [[390, 389], [401, 407]], [[378, 390], [387, 407]], [[368, 391], [377, 408]], [[300, 387], [312, 407]], [[242, 387], [252, 406]], [[169, 391], [182, 416]], [[41, 398], [63, 433]]]
Frame_800=[800,[[729, 468], [755, 421]], [[597, 402], [614, 431]], [[550, 397], [563, 417]], [[521, 396], [531, 410]], [[482, 392], [492, 407]], [[457, 391], [470, 407]], [[443, 392], [452, 408]], [[410, 390], [420, 406]], [[390, 389], [401, 407]], [[378, 390], [387, 407]], [[368, 391], [377, 408]], [[300, 387], [312, 407]], [[242, 387], [252, 406]], [[169, 391], [182, 416]], [[41, 398], [63, 433]]]

cone_coords=[Frame_150,Frame_200,Frame_250,Frame_350,Frame_400,Frame_500,Frame_600,Frame_700,Frame_800]


run_number = 1

for i in range(len(cone_coords)):
    frame_number=cone_coords[i][0]
    depth_data_path = os.path.join(data_path,"Run{}".format(run_number))
    depth_arr=np.load(os.path.join(depth_data_path,"depth_{}.npy".format(frame_number)))
    coords=cone_coords[i][1:]
    #print(coords)
    #Make image from point cloud and display with boxes
    try:
        print("frame number: {}".format(frame_number))
        points=display_frame_from_point_cloud_with_boxes_and_return_cone_pos(depth_arr,coords)
        plot_points_with_delanay(points)
    except:
        print("\n---Error in frame {}---\n".format(frame_number))


