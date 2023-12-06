import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import DelanayTriangles_copy_2 as DT

import keyboard



#Points from racetrack 1:
yellow_points=[[3.0, 0.0], [6.0, 0.0], [9.0, 0.0], [12.0, 0.0], [14.968847511054276, -0.37305093947226275], [17.753106463250436, -1.4690092573155287], [20.17966512028001, -3.2197335735141515], [22.097651817694757, -5.516372329582325], [23.387815432267033, -8.216131651256777], [23.969939839248653, -11.151153579987566], [24.0, -15.0], [24.734504628657763, -18.87655323162522], [26.758186164791162, -21.04882590884738], [29.575576789993782, -21.984969919624326], [32.49688101928285, -21.45578456095409], [34.806861693281604, -19.59083286462374], [36.664216154692454, -15.233042997596288], [37.69027658466946, -12.413965135238563], [38.71633701464647, -9.594887272880838], [39.74239744462348, -6.775809410523113], [41.41644517267064, -4.3239489438339325], [44.06104479140185, -2.974820187747273], [47.02870608051554, -3.0587369142634477], [49.59284205563091, -4.555153382033144], [51.905182657112405, -7.269337941714823], [53.83354548617202, -9.567471271071756], [55.761908315231636, -11.86560460042869], [57.69027114429125, -14.163737929785622], [59.20363258912161, -16.737977979647624], [59.79141630610384, -19.66568783533057], [59.500156210803205, -23.72181219596864], [59.238688982560234, -26.710396290243875], [58.97722175431726, -29.69898038451911], [58.71575452607429, -32.68756447879435], [57.91105956262807, -37.60499049322057], [56.57660754008259, -40.28313543830076], [54.621056736391054, -42.54787462891757], [52.165993828970606, -44.25839755128446], [49.3640627385007, -45.30835217487179], [46.389473969254055, -45.632457406557485], [43.42717304371586, -45.21056195281351], [40.66134148576471, -44.06889722953622], [37.24673329439849, -41.74318018442071], [34.789277161531516, -40.02245087536757], [30.373344065372752, -37.82704581920054], [27.433186417268292, -38.23878426553027], [25.050353262478897, -40.00970541011282], [22.78615577396048, -42.56671592971564], [20.291413949764475, -44.21883147589923], [17.465487680068122, -45.20237774607383], [14.484079373169688, -45.45620259688893], [11.532558557375648, -44.964524428501534], [8.794436501553319, -43.75791340801747], [5.6244042885658665, -41.574929636495376], [3.1669481556988903, -39.854200327442236], [0.7094920228319141, -38.133471018389095], [-1.747964110035062, -36.412741709335954], [-3.6830040228380994, -34.16114758966155], [-4.30168958333295, -31.25748030136416], [-3.4525449888972184, -28.412658865555976], [-1.5006884486096055, -25.598914692628764], [0.22004086044353066, -23.141458559761787], [2.415445916610563, -18.72552546360303], [2.0037074702808337, -15.78536781549857], [0.23278632569828117, -13.402534660709176], [-2.4637342579091657, -12.160426659738839], [-5.826374857763842, -11.102109914278875], [-7.814005914255694, -8.896803058722885], [-8.50103584189221, -6.00854312886461], [-7.719255753381579, -3.1444768871199336], [-5.660072679924978, -1.0058276372701584], [-2.8276464544249413, -0.11621130367865895], [0.49815038038356185, -0.1073594759765825]]
blue_points=[[3.0, -3.0], [6.0, -3.0], [9.0, -3.0], [12.0, -3.0], [14.94475227116537, -3.4953874831673613], [17.56532822762763, -4.927014653007467], [19.57323886327107, -7.1372792471867434], [20.747441112269815, -9.882861840273096], [21.0, -15.0], [21.495387483167363, -18.94475227116537], [22.92701465300747, -21.56532822762763], [25.137279247186743, -23.57323886327107], [27.882861840273097, -24.747441112269815], [30.861511932129382, -24.958671619765884], [33.74532152892428, -24.183676841431136], [36.216823257748885, -22.507772935644923], [38.00393911391737, -20.115453639722308], [39.48329401705018, -16.259103427573294], [40.509354447027185, -13.440025565215569], [41.53541487700419, -10.620947702857844], [42.5614753069812, -7.801869840500119], [44.72079898037038, -5.901375229112199], [47.48669761248492, -6.6915418262551345], [49.60704932775547, -9.197700770774441], [51.535412156815084, -11.495834100131376], [53.4637749858747, -13.793967429488308], [55.392137814934316, -16.09210075884524], [56.678487401598346, -18.767798851826303], [56.511572116527965, -23.460344967725664], [56.250104888284994, -26.4489290620009], [55.98863766004202, -29.437513156276136], [55.72717043179905, -32.42609725055138], [54.80270449067011, -37.308857382420086], [53.14812685051918, -39.79468672686936], [50.771272037597456, -41.60231940396483], [47.933798746044396, -42.532760168729304], [44.94807336529489, -42.483580419231004], [42.14278278019303, -41.46019416265365], [38.96746260345162, -39.28572405155373], [36.51000647058465, -37.56499474250059], [32.17535995194885, -35.13459902571854], [29.207562189737978, -34.80421873025788], [26.295022398478814, -35.46307126409569], [23.758370746352824, -37.03862611645056], [20.242307873062916, -40.75607959688174], [17.566486931933873, -42.08158213857216], [14.604253944288615, -42.45861055118685], [11.681709608861572, -41.84565924441297], [7.345133597619003, -39.117473503628396], [4.887677464752026, -37.396744194575255], [2.4302213318850505, -35.676014885522115], [-0.027234800981925877, -33.955285576468974], [-1.3040975376308697, -31.377654872483088], [0.9567676842573716, -27.3196440016819], [2.6774969933105077, -24.862187868814924], [5.107892710092555, -20.527541350179128], [5.438273005553224, -17.559743587968256], [4.779420471715415, -14.647203796709091], [3.2038656193605433, -12.110552144583105], [0.8850551491330942, -10.229038737727857], [-1.9217421205695824, -9.209792063069495], [-5.157927766936066, -7.502081267349734], [-5.110552686499008, -4.625918181548259], [-2.664748037020689, -3.111785389827621], [0.4981503803835628, -3.1073594759765824]]


#yellow_points
yellow_points_np=np.array(yellow_points).reshape(-1,2)
x_yellow=yellow_points_np[:,0]
y_yellow=yellow_points_np[:,1]

#blue_points
blue_points_np=np.array(blue_points).reshape(-1,2)
x_blue=blue_points_np[:,0]
y_blue=blue_points_np[:,1]



#plot
fig_1, ax_1 = plt.subplots()
ax_1.set_aspect('equal') # Set the aspect ratio to 1

ax_1.scatter(x_yellow, y_yellow, c='gold', marker='o') # Create a scatter plot object
ax_1.scatter(x_blue, y_blue, c='b', marker='o') # Create a scatter plot object



def transform_points(list_of_points, x_shift, y_shift, theta):
    #list_of_points: list of points to be transformed
    #x_shift: shift in x direction
    #y_shift: shift in y direction
    #theta: rotation angle in degrees
    #returns: list of transformed points
    list_of_points_np=np.array(list_of_points).reshape(-1,2)
    x=list_of_points_np[:,0]
    y=list_of_points_np[:,1]

    #shift
    x=x-x_shift
    y=y-y_shift

    #rotate
    theta=np.deg2rad(theta)
    x_rot=x*np.cos(theta)-y*np.sin(theta)
    y_rot=x*np.sin(theta)+y*np.cos(theta)

    transform_points_np=np.array([x_rot,y_rot]).T

    return transform_points_np

#oriantation=90 #degrees
#pos=[0,0] #position of the car x,y in meters


def car_view(pos,oriantation):
    global view_depth, view_width
    yellow_points_tran=transform_points(yellow_points_np, pos[0], pos[1], oriantation)
    blue_points_tran=transform_points(blue_points_np, pos[0], pos[1], oriantation)

    #only the points that are in front of the car within in 20 meters and 5 meters to the left and right
    #view_depth=20 #meters from the car
    #view_width=10 #meters from the center of the car

    yellow_points_tran=yellow_points_tran[np.logical_and(np.logical_and(np.abs(yellow_points_tran[:,0])<view_width, yellow_points_tran[:,1]<view_depth), yellow_points_tran[:,1]>0)]
    blue_points_tran=blue_points_tran[np.logical_and(np.logical_and(np.abs(blue_points_tran[:,0])<view_width, blue_points_tran[:,1]<view_depth), blue_points_tran[:,1]>0)]
    #midpoints_tran=midpoints_tran[np.logical_and(np.logical_and(np.abs(midpoints_tran[:,0])<view_width, midpoints_tran[:,1]<view_depth), midpoints_tran[:,1]>0)]

    return yellow_points_tran, blue_points_tran#, midpoints_tran

def get_middelpoint_with_DT(yel_points, blu_points):
    #generate the point array
    point_array=[]
    for i in range(yel_points.shape[0]):
        point_array.append([yel_points[i,0],yel_points[i,1],'yellow'])

    for i in range(blu_points.shape[0]):
        point_array.append([blu_points[i,0],blu_points[i,1],'blue'])
    
    # convert the point array to a NumPy array, without color information
    point_array_without_color = DT.remove_Colors(point_array)


    # Use the filter function to remove the triangles that are made by three points of the same color
    tri = DT.delaunay_triangles_filtered(point_array, point_array_without_color)

    # Find the midpoints of the triangles that are made by two points of different color
    midpoints = DT.find_midpoints(tri, point_array)

    midpoints_np=np.array(midpoints).reshape(-1,2)
    return midpoints_np



#The car's field of view
view_depth=20 #meters from the car
view_width=10 #meters from the center of the car

#plot the car
car_pos=[0.2,0] #position of the car's back right coner x,y in meters
car_width=0.7
car_length=0.4
car_theta=90
car_speed=2.5 #meters per second

def get_stering_angle(point_pos):
    global car_length
    #point_pos: position of the point the car shall be heading towards
    #returns: stering angle in degrees
    x_1=point_pos[0]
    y_1=point_pos[1]
    radius=(x_1**2+y_1**2)/(2*x_1)
    stering_angle=np.arctan(car_length/radius)
    #print(f"stering_angle={np.rad2deg(stering_angle)}, radius={radius}, x_1={x_1}, y_1={y_1}")
    return stering_angle

def update_pos_and_oriantation(pos, oriantation, stering_angle, speed, dt):
    global start_oriantation, car_length
    #pos: position of the car's back right coner x,y in meters
    #oriantation: oriantation of the car in degrees
    #stering_angle: stering angle in radians
    #speed: speed of the car in meters per second
    #dt: time step in seconds
    #returns: new position and oriantation of the car
    x=pos[0]
    y=pos[1]
    theta=np.deg2rad(oriantation)
    theta_new=theta+speed*np.tan(stering_angle)/car_length*dt
    theta=theta_new
    stering_angle=stering_angle
    x_new=x+speed*np.cos(np.deg2rad(start_oriantation)-theta)*dt
    y_new=y+speed*np.sin(np.deg2rad(start_oriantation)-theta)*dt
    theta_new=theta+speed*np.tan(stering_angle)/car_length*dt
    return [x_new, y_new], np.rad2deg(theta_new)

#plot
fig_2, ax_2 = plt.subplots()
ax_2.set_aspect('equal') # Set the aspect ratio to 1
lim_x=plt.xlim([-view_width-1, view_width+1]) # Set the x axis limits
lim_y=plt.ylim([-1, view_depth+1]) # Set the y axis limits

#rotat the rectangle 45 degrees
car=ax_2.add_patch(Rectangle(car_pos, car_width, car_length, edgecolor='red', facecolor='red', lw=4, angle=car_theta))


start_oriantation=90
oriantation_=start_oriantation
pos_=[0,(blue_points_np[1,1]-yellow_points_np[1,1])/2]
dt=0.1#dt=0.03 #time step in seconds
stering_angle=0 #degrees
diff_angle=0
num_midtpoints_too_close=0
number_of_midpoints=4
pause=True
integral_error=0
counter=0
route_taken=[]

for i in range(1100):
    #run if key 'q' is not pressed
    if keyboard.is_pressed('z'):
        print("The key 'z' was pressed and the program is now terminated!")
        #close all figures
        plt.close('all')
        break
    plt.xlim([-view_width-1, view_width+1]) # Set the x axis limits
    plt.ylim([-1, view_depth+1]) # Set the y axis limits


    pos_,oriantation_=update_pos_and_oriantation(pos_,oriantation_,stering_angle,car_speed,dt)

    #print(f"pos_={pos_}, oriantation_={oriantation_}")

    #pos_=[pos_[0]+np.cos(np.deg2rad(90-oriantation_))*1, pos_[1]+np.sin(np.deg2rad(90-oriantation_))*1]
    
    #transformed points
    yellow_points_tran, blue_points_tran=car_view(pos_,oriantation_)
    midpoints=get_middelpoint_with_DT(yellow_points_tran[0:,:], blue_points_tran[0:,:])
    midpoints=midpoints[num_midtpoints_too_close:]
    if (midpoints[0,0]**2+midpoints[0,1]**2)<(1.2**2):
        mid_points_next=midpoints[1:number_of_midpoints+1,:]
    else:
        mid_points_next=midpoints[:number_of_midpoints,:]
    #mid_points_next=midpoints[:number_of_midpoints,:]
    """
    j=0
    k=0
    for i in range(number_of_midpoints):
        j=i
        while True:
            k+=1
            if midpoints[i,0]<abs(midpoints[i,1]):
                mid_points_next[i,:]=midpoints[i,:]
                j=1
            if k>number_of_midpoints*1.5:
                break
            j+=1
    """
    angles_x=[]
    
    gain=3/5
    sum_gain=0
    for i in range(len(mid_points_next)):
        
        angles_x.append(get_stering_angle(mid_points_next[i,:])*gain)
        sum_gain+=gain
        if gain==3/5:
            gain=2/5
        if i<(len(mid_points_next)-2):
            gain=gain/2
    
        
        """
        for j in range(gain):
            angles_x.append(get_stering_angle(mid_points_next[i,:]))
        gain-=1
    #angles_x[:]=get_stering_angle(mid_points_next[:,:])
    stering_angle=np.mean(angles_x)"""
    stering_angle_new=np.sum(angles_x)
    if stering_angle_new>np.deg2rad(25):
        stering_angle_new=np.deg2rad(25)
    if stering_angle_new<np.deg2rad(-25):
        stering_angle_new=np.deg2rad(-25)

    #PID
    #kp=0.05
    #ki=0.010
    #kd=0.005
    kp=0.4
    ki=0.001
    kd=0.001
    angle_diff=stering_angle_new-stering_angle
    integral_error+=angle_diff
    stering_angle=angle_diff*kp+integral_error*ki+(diff_angle-stering_angle)*kd
    #stering_angle+=(stering_angle_new-stering_angle)
    #stering_angle=stering_angle_new*0.05
    #print(f"angles_x={angles_x}, sum={np.sum(angles_x)}, sum gain={sum_gain}")
    #print(f"stering_angle={stering_angle}")
 
    ax_2.add_patch(Rectangle(car_pos, car_width, car_length, edgecolor='red', facecolor='red', lw=4, angle=car_theta))

    if counter%int(1/dt)==0:
        ax_1.scatter(pos_[0],pos_[1],c='r', marker='.')
    counter+=1
    route_taken.append(pos_)


    ax_2.scatter(yellow_points_tran[:,0], yellow_points_tran[:,1], c='gold', marker='o') # Create a scatter plot object
    ax_2.scatter(blue_points_tran[:,0], blue_points_tran[:,1], c='b', marker='o') # Create a scatter plot object
    ax_2.scatter(midpoints[:,0], midpoints[:,1], c='r', marker='o') # Create a scatter plot object
    
    if keyboard.is_pressed('p'):
        pause=True
        plt.pause(0.1)

    #pause the program if key 'p' is pressed
    if pause==True:
        print("The program is now paused until the key 'p' is pressed!")
        while True:
            if keyboard.is_pressed('p'):
                print("The key 'p' was pressed and the program is now continuing!")
                break
            else:
                plt.pause(0.1)
        pause=False
    
    plt.pause(dt)
    #plt.pause(8)
    ax_2.clear()

#show route taken in the end
route_taken_np=np.array(route_taken).reshape(-1,2)
ax_1.scatter(route_taken_np[:,0], route_taken_np[:,1], c='r', marker='.')
plt.show()




