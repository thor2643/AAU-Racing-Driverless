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

x_slice=int(arr_1.shape[1]/2)

arr_1_indexes=np.argwhere(arr_1[200:,x_slice]>-1)
arr_1_values=arr_1[200:,x_slice].flatten()
arr_1_values_modsat=1/arr_1_values[:]
print(arr_1_values)
print(1/arr_1_values_modsat)

print(arr_1_values_modsat)



LS=LinearRegression()
model=LS.fit(arr_1_indexes,arr_1_values_modsat)


y_pred = LS.predict(arr_1_indexes)
#display the linear regression and the data points
plt.scatter(arr_1_indexes, arr_1_values_modsat,color='g')
plt.plot(arr_1_indexes, y_pred,color='k')
plt.title('Linear Regression på x_slice midten af billedet og x_slice data points')
plt.show()
arr_2_values=arr_1[200:,0].flatten()
arr_2_values_modsat=1/arr_2_values[:]
plt.scatter(arr_1_indexes, arr_2_values_modsat,color='g')
plt.plot(arr_1_indexes, y_pred,color='k')
plt.title('Linear Regression på x_slice midten af billedet og 0X data points')
plt.show()

arr_3_indexes=np.argwhere(arr_1[200,:]>-1)
arr_3_values=arr_1[200,:].flatten()
arr_3_values_modsat=arr_3_values[:]
LR=LinearRegression()
model_2=LR.fit(arr_3_indexes,arr_3_values_modsat)
y_pred_3 = LR.predict(arr_3_indexes)

plt.scatter(arr_3_indexes, arr_3_values_modsat,color='g')
plt.plot(arr_3_indexes, y_pred_3,color='k')
plt.title('Linear Regression på y_slice midten af billedet og y data points')
plt.show()


print(abs(arr_1_values[-1]-(1/arr_2_values_modsat[-1])))

arr_3=arr_1.copy()

for y, row in enumerate(arr_3):
    for x, pixel in enumerate(row):
        if abs(pixel-(1/model.predict([[y-200]])))<500:
            arr_3[y,x]=0
        if pixel > 5800 or pixel < 1:
            arr_3[y,x]=0


def map_arr(arr,from_no,to_no):
    map_arr=interp1d([np.amin(arr),np.amax(arr)],[from_no,to_no])
    return np.uint8(map_arr(arr))

depth_img=cv2.applyColorMap(cv2.convertScaleAbs(depth_arr_crop, alpha=0.03), cv2.COLORMAP_JET)

cv2.imshow("depth_arr_crop jet",depth_img)
cv2.imshow("arr_1",map_arr(arr_1[200:,:],0,255))
cv2.imshow("arr_3",map_arr(arr_3,0,255))
cv2.waitKey(0)

