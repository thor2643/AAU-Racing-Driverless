import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from mpl_toolkits.mplot3d import Axes3D



data = [[[5, 184.0, 175.8, False, 1.5, 0.479, 24, 16, 336.5, 304, 376, 328, 392, 0.5468053491827637], [False]], 
         [[18, 34.5, 54.2, False, 1.625, 0.332, 13, 8, 72.5, 411, 368, 424, 376, 0.47586206896551725], [True]], 
         [[30, 175.5, 213.3, False, 2.5, 0.274, 40, 16, 441.0, 336, 368, 376, 384, 0.3979591836734694], [False]], 
         [[50, 257.5, 180.4, False, 1.0, 0.447, 24, 24, 418.5, 216, 248, 240, 272, 0.6152927120669056], [False]], 
         [[62, 1.0, 43.1, False, 0.455, 0.018, 5, 11, 29.5, 216, 229, 221, 240, 0.03389830508474576], [True]], 
         [[64, 96.0, 83.4, False, 0.812, 0.462, 13, 16, 162.5, 219, 224, 232, 240, 0.5907692307692308], [False]], 
         [[81, 100.0, 116.4, False, 0.458, 0.379, 11, 24, 197.0, 309, 200, 320, 224, 0.5076142131979695], [False]], 
         [[88, 43.5, 131.6, False, 1.143, 0.194, 16, 14, 178.0, 384, 194, 400, 208, 0.2443820224719101], [True]], 
         [[108, 23.0, 39.5, False, 1.0, 0.359, 8, 8, 42.5, 80, 184, 88, 192, 0.5411764705882353], [True]], 
         [[115, 11.5, 31.6, False, 0.778, 0.183, 7, 9, 31.5, 168, 183, 175, 192, 0.36507936507936506], [True]], 
         [[122, 113.0, 130.2, False, 1.438, 0.307, 23, 16, 251.0, 480, 176, 503, 192, 0.450199203187251], [False]], 
         [[137, 14.0, 41.1, False, 1.0, 0.219, 8, 8, 39.5, 320, 176, 328, 184, 0.35443037974683544], [False]], 
         [[140, 45.5, 69.5, False, 2.0, 0.355, 16, 8, 88.0, 248, 176, 264, 184, 0.5170454545454546], [False]], 
         [[149, 23.5, 28.4, False, 1.143, 0.42, 8, 7, 35.0, 168, 176, 176, 183, 0.6714285714285714], [True]], 
         [[156, 110.5, 259.3, False, 0.533, 0.23, 16, 30, 397.5, 440, 170, 456, 200, 0.2779874213836478], [False]], 
         [[167, 25.5, 28.4, False, 0.875, 0.455, 7, 8, 27.0, 448, 181, 455, 189, 0.9444444444444444], [True]],
         [[170, 44.5, 205.2, False, 0.615, 0.107, 16, 26, 297.5, 384, 168, 400, 194, 0.1495798319327731], [False]], 
         [[198, 31.0, 29.0, False, 1.0, 0.484, 8, 8, 40.5, 520, 160, 528, 168, 0.7654320987654321], [False]], 
         [[205, 13.0, 52.3, False, 1.2, 0.108, 12, 10, 51.5, 443, 160, 455, 170, 0.2524271844660194], [False]], 
         [[213, 14.0, 44.3, False, 1.25, 0.175, 10, 8, 47.0, 244, 160, 254, 168, 0.2978723404255319], [True]], 
         [[219, 17.5, 38.0, False, 0.875, 0.312, 7, 8, 35.5, 193, 160, 200, 168, 0.49295774647887325], [True]], 
         [[226, 6.5, 37.7, False, 1.0, 0.102, 8, 8, 25.5, 144, 160, 152, 168, 0.2549019607843137], [False]], 
         [[229, 30.0, 28.6, False, 1.0, 0.469, 8, 8, 37.5, 64, 160, 72, 168, 0.8], [False]], 
         [[242, 18.5, 77.8, False, 1.364, 0.112, 15, 11, 94.0, 360, 125, 375, 136, 0.19680851063829788], [False]],
         [[246, 40.0, 127.3, False, 0.812, 0.192, 13, 16, 157.5, 371, 120, 384, 136, 0.25396825396825395], [True]], 
         [[255, 91.5, 113.4, False, 1.143, 0.408, 16, 14, 174.5, 464, 114, 480, 128, 0.5243553008595988], [True]], 
         [[278, 31.0, 192.6, False, 4.0, 0.121, 32, 8, 214.0, 296, 104, 328, 112, 0.14485981308411214], [False]], 
         [[287, 94.5, 212.5, False, 5.5, 0.268, 44, 8, 283.0, 251, 104, 295, 112, 0.3339222614840989], [False]], 
         [[296, 23.5, 26.4, False, 1.429, 0.336, 10, 7, 31.5, 252, 105, 262, 112, 0.746031746031746], [True]]]

features = [feature[0] for feature in data]
y = [feature[1][0] for feature in data]


StdScaler = StandardScaler()
Pca = PCA(n_components=3)

#Scale the data to 0 mean and unit variance
data_scaled = StdScaler.fit_transform(features)

#Perform pca analysis of the scaled data
data_pca = Pca.fit_transform(data_scaled)

X_train, X_test, y_train, y_test = train_test_split(data_pca, y, test_size=0.20)


model = GaussianNB()

model.fit(X_train, y_train)

predicted = model.predict(X_test)

accuracy = accuracy_score(predicted, y_test)
f1 = f1_score(predicted, y_test, average="weighted")

print(accuracy)
print(f1)



"""
nums = np.arange(14)
var_ratio = []

for num in nums:
  pca = PCA(n_components=num)
  pca.fit(data_scaled)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(4,2),dpi=150)
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')



# Adding legend
plt.legend()

# Displaying the plot
plt.show()


# Creating a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pca[:10, 0], data_pca[:10, 1], data_pca[:10, 2], c='r', label='Class 1')
ax.scatter(data_pca[10:20, 0], data_pca[10:20, 1], data_pca[10:20, 2], c='b', label='Class 2')
ax.scatter(data_pca[20:30, 0], data_pca[20:30, 1], data_pca[20:30, 2], c='g', label='Class 3')
ax.scatter(data_pca[30:40, 0], data_pca[30:40, 1], data_pca[30:40, 2], c='#2ca02c', label='Class 4')

# Adding labels and title
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title('3D Scatter Plot of 4 Classes')


# Displaying the plot
plt.show()
"""