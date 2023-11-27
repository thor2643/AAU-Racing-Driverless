import cv2
import numpy as np
import os
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path_="C:\\Users\\3103e\\Documents\\GitHub\\AAU-Racing-Driverless\\Depth_cone_detection"


depth_arr=np.load(os.path.join(path_,"depth_5365.npy"))
depth_arr_crop=depth_arr[100:400,:]

arr_1=depth_arr_crop.copy()
arr_2=arr_1[200:,:]
print(arr_2)
"""
for y, row in enumerate(arr_2):
    for x, value in enumerate(row):
        if value==0:
            print(f"y: {y}, x: {x} arr_1[y,x]: {arr_2[y,x]}")
print(arr_2[0,0])
"""
arr_1_indexes=np.argwhere(arr_2>199)
#arr_1_indexes=(arr_2>0).nonzero()
print("hej")
print(arr_1_indexes)
#print(f"arr_1_indexes[0][0]: {arr_1_indexes[0][0]}, arr_1_indexes[0][1]: {arr_1_indexes[0][1]}")
print(arr_2[arr_1_indexes[0][0],arr_1_indexes[0][1]])

arr_1_values_list=np.array([])

for i,index in enumerate(arr_1_indexes):
    arr_1_values_list=np.append(arr_1_values_list,[arr_2[index[0],index[1]]])
print(arr_1_indexes.shape)
print(arr_1_values_list.shape)

arr_1_values_modsat=1/arr_1_values_list[:]


LS=LinearRegression()
model=LS.fit(arr_1_indexes,arr_1_values_modsat)

print(f"model.coef_: {model.coef_}, model.intercept_: {model.intercept_}")
print(model.predict([[1,1]]))
print(arr_2[1,1])
print(1/model.predict([[1,1]]))
print(abs(arr_2[1,1]-1/model.predict([[1,1]])))



"""
for y, row in enumerate(arr_1):
    for x, value in enumerate(row):
        if abs(arr_2[y,x]-1/(model.predict([[y,x]]))):
"""

df2=pd.DataFrame(arr_1_indexes,columns=['Y_cor','X_cor'])
df2['Depth']=pd.Series(arr_1_values_modsat)

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

def map_arr(arr,from_no,to_no):
    map_arr=interp1d([np.amin(arr),np.amax(arr)],[from_no,to_no])
    return np.uint8(map_arr(arr))

depth_img=cv2.applyColorMap(cv2.convertScaleAbs(depth_arr_crop, alpha=0.03), cv2.COLORMAP_JET)

arr_3=arr_1.copy()

print(arr_3[200,200]-(1/(model.predict([[200-200,200]]))))

for y, row in enumerate(arr_3):
    for x, pixel in enumerate(row):
        if abs(pixel-(1/model.predict([[y-200,x]])))<500:
            arr_3[y,x]=0
        if pixel > 5300 or pixel < 1:
            arr_3[y,x]=0
        

cv2.imshow("depth_arr_crop jet",depth_img)
cv2.imshow("arr_1",map_arr(arr_1[200:,:],0,255))
cv2.imshow("arr_2",map_arr(arr_2,0,255))
cv2.imshow("arr_3",map_arr(arr_3,0,255))
cv2.waitKey(0)

