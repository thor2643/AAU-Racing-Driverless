import numpy as np
import matplotlib.pyplot as plt

ground_truth_points = np.array([[-4760, 4400], [-4000, 7000], [-2240, 2640], [1660, 8880], [5900, 5660], [3520, 4920], [5000, 2000], [1440, 3340], [-1720, 9880], [0, 6000]])
ground_truth_points[:,1] = ground_truth_points[:,1] + 25
test1_list = np.array([[-6474.522619870042, 826.0059592951886], [-5717.648251339817, 3074.8272266829454], [-2814.562960434822, 2065.264230491675], [4271.691417490318, 3305.06481536135], [527.9110599415193, 3631.832858597904], [2102.265126128572, 5692.196881649792], [-1534.1904676645468, 5835.7018094593595], [-5687.803032037682, 5741.7436958421395], [5020.6518524696885, 8602.01458239936], [-744.7189096150678, 9001.245344154428]]) #[-1534.1904676645468, 5835.7018094593595]
test2_list = np.array([[-747.6872312211508, 9037.122761380906], [-5689.2105556783745, 5743.164567828242], [-1535.4617557550873, 5840.537492098952], [2103.3044793550694, 5695.011085777173], [528.1987498924411, 3633.8120590658045], [-2841.197323477437, 2009.4769889363376], [-5722.9325842892995, 3077.6690263379205], [4272.482324992167, 3305.6767510798027]]) #[-1535.4617557550873, 5840.537492098952]
test3_list = np.array([[2103.3044793550694, 5695.011085777173], [-746.7802440637365, 9026.160217228371], [-1534.698982900763, 5837.636082515196], [527.9110599415193, 3631.832858597904], [4264.5732499736705, 3299.557393895279], [-2839.56445145245, 2008.3221171036155], [-5722.051862131052, 3077.195393062091], [-5690.618079319067, 5744.585439814346]])
test4_list = np.array([[-5797.168745050687, 5651.37598655473], [-2612.569047649931, 9276.111899457743], [2008.8729159981465, 5745.95497784042], [465.58540282686266, 3649.4206708290767], [4204.77830352544, 3372.230629450146], [-2867.570911383166, 1953.954213432116], [-5797.9950361758065, 2988.7760974152457]])
test5_list = np.array([[], [], [], [], [], [], [], [], [], []])
test6_list = np.array([[], [], [], [], [], [], [], [], [], []])
test7_list = np.array([[], [], [], [], [], [], [], [], [], []])
test8_list = np.array([[], [], [], [], [], [], [], [], [], []])
test9_list = np.array([[], [], [], [], [], [], [], [], [], []])
test10_list = np.array([[], [], [], [], [], [], [], [], [], []])
test11_list = np.array([[], [], [], [], [], [], [], [], [], []])
test12_list = np.array([[], [], [], [], [], [], [], [], [], []])

test_lists = [test1_list, test2_list, test3_list, test4_list]

def rotate_points(ref_point, list_of_points): #theta=np.deg2rad(-14.729600000031802)
    theta = -1*( np.arctan2(ref_point[1], ref_point[0]) - np.deg2rad(90))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_points = rotation_matrix@list_of_points.T
    print(f"theta={np.rad2deg(theta)}")
    return rotated_points.T

def create_point_cloud(points):
    x_truth = ground_truth_points[:, 0]
    y_truth = ground_truth_points[:, 1]
    x = points[:, 0]
    y = points[:, 1]

    plt.scatter(x_truth, y_truth, c='r',marker='o', s=20)
    plt.scatter(x, y, marker='x', s=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Point Cloud')
    plt.show()

def find_closest_point(ground_truth_point, point_list, threshold=200):
    closest_point = []
    for point in point_list:
        euclidean_dist = np.sqrt((ground_truth_point[0] - point[0])**2 + (ground_truth_point[1] - point[1])**2)
        if euclidean_dist < threshold and closest_point == []:
            closest_point.append(point)
    return closest_point

def find_euclidean_dist_deviation(ground_truth_point_list, point_list):
    deviation_list = []
    for i in range(len(ground_truth_point_list)):
        point = []
        ground_truth_point = ground_truth_point_list[i]
        point = find_closest_point(ground_truth_point, point_list)
        if point == []:
            continue
        euclidean_dist_deviation = np.sqrt((ground_truth_point[0] - point[0][0])**2 + (ground_truth_point[1] - point[0][1])**2)
        deviation_list.append(euclidean_dist_deviation)
    return deviation_list

def find_closest_point_to_ref(ref_point, point_list):
    closest_dist = 0
    for i in range(len(point_list)):
        euclidean_dist = np.sqrt((ref_point[0] - point_list[i][0])**2 + (ref_point[1] - point_list[i][1])**2)
        if closest_dist == 0:
            closest_dist = euclidean_dist
            index = i
        elif euclidean_dist < closest_dist:
            closest_dist = euclidean_dist
            index = i
    closest_point = np.array(point_list[index])
    print(f'closest point: {closest_point}')
    return closest_point

    

# Example usage
"""
rotated_point_list1 = rotate_points([-1534.1904676645468, 5835.7018094593595], test1_list)
rotated_point_list2 = rotate_points([-1535.4617557550873, 5840.537492098952], test2_list)

create_point_cloud(rotated_point_list1)
create_point_cloud(rotated_point_list2)
"""

# main()
create_point_cloud(test4_list)

for i in range(len(test_lists)):
    closest_point_to_ref = find_closest_point_to_ref([1440, 3340], test_lists[i])
    print(closest_point_to_ref)

    rotated_point_list = rotate_points(closest_point_to_ref, test_lists[i])

    deviation = find_euclidean_dist_deviation(ground_truth_points, rotated_point_list)

    avrg_deviation = np.mean(deviation)
    print(f'average deviation: {avrg_deviation}')
    create_point_cloud(rotated_point_list)

