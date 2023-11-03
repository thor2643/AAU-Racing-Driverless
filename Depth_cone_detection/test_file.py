import cv2
import numpy as np
import os
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


print("test")
path_="C:\\Users\\3103e\\Documents\\GitHub\\AAU-Racing-Driverless\\Depth_cone_detection"

#arr=np.load("depth_5365.npy")
arr=np.load(os.path.join(path_,"depth_5365.npy"))[100:400,:]

#print 10 x10 matrix from the "deoth_5365.npy" file
print(arr.shape)

#findes max and min value in the matrix arr
arr_3=arr.copy()
arr_2=arr.copy()

#list of the indexes in arr_2
arr_2_indexes=np.argwhere(arr_2[250:,:]>-1)
print(arr_2_indexes.shape)

#list of the values in arr_2
arr_2_values=arr_2[250:,:].flatten()
print(arr_2_values.shape)

print(arr_2[0,0])

print("hej")



LS=LinearRegression()
model=LS.fit(arr_2_indexes,arr_2_values)

df2=pd.DataFrame(arr_2_indexes,columns=['Y_cor','X_cor'])
df2['Depth']=pd.Series(arr_2_values)

x_surf, y_surf = np.meshgrid(np.linspace(df2.Y_cor.min(), df2.Y_cor.max(), 100),np.linspace(df2.X_cor.min(), df2.X_cor.max(), 100))

onlyX = pd.DataFrame({'Y_cor': x_surf.ravel(), 'X_cor': y_surf.ravel()})
fittedY=LS.predict(onlyX)

fittedY=np.array(fittedY)

fig = plt.figure(figsize=(20,10))
### Set figure size
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['Y_cor'],df2['X_cor'],df2['Depth'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('Y_cor')
ax.set_ylabel('X_cor')
ax.set_zlabel('Depth')
plt.show()

print(model.predict([[0,100]]))

print(arr_2[0,100])

depth_img=cv2.applyColorMap(cv2.convertScaleAbs(arr, alpha=0.03), cv2.COLORMAP_JET)

for y, row in enumerate(arr_2):
    for x, pixel in enumerate(row):
        if pixel > 9000 or pixel < 1:
            arr_2[y,x]=9000

for y, row in enumerate(arr_2[240:,:]):
    for x, pixel in enumerate(row):
        #print(f"pixel: {pixel}, model.predict: {model.predict([y,x])} forsekl: {abs(pixel-(model.predict([[y+200,x]])))}")
        if abs(pixel-(np.uint8(model.predict([[y,x]]))))<1000:
            arr_3[y+240,x]=0

for y, row in enumerate(arr_3):
    for x, pixel in enumerate(row):
        #print(f"pixel: {pixel}, model.predict: {model.predict([y,x])} forsekl: {abs(pixel-(model.predict([[y+200,x]])))}")
        if pixel> 9000:
            arr_3[y,x]=0

print(arr_2_values[-20])
print(model.predict([arr_2_indexes[-20]]))
print(f"forsekl:{abs(arr_2[-20,-20]-model.predict([arr_2_indexes[-20]]))}")
#print(arr_2)

#linear regression of the values in arr_2

def map_arr(arr,from_no,to_no):
    map_arr=interp1d([np.amin(arr),np.amax(arr)],[from_no,to_no])
    return np.uint8(map_arr(arr))


"""
arr_uint8=np.zeros(arr.shape,dtype=np.uint8)
arr_uint8=(arr-min_val)/(max_val-min_val)*255
arr_uint8=np.array(arr_uint8,dtype=np.uint8)
#invers image arr_uint8
#arr_uint8=255-arr_uint8"""

cv2.imshow("test",map_arr(arr,255,0))
cv2.imshow("test 2",map_arr(arr_2,255,0))
cv2.imshow("test 3",depth_img)
cv2.imshow("arr_3",map_arr(arr_3,0,255))
cv2.imshow("arr_3 jet",cv2.applyColorMap(cv2.convertScaleAbs(arr_3, alpha=0.03), cv2.COLORMAP_JET))
cv2.waitKey(0)