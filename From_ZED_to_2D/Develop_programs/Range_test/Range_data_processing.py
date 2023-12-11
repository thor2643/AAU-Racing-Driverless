import numpy as np
import matplotlib.pyplot as plt

ground_truth_points = np.array([[-4760, 4400], [-4000, 7000], [-2240, 2640], [1660, 8880], [5900, 5660], [3520, 4920], [5000, 2000], [1440, 3340], [-1720, 9880], [0, 6000]])
ground_truth_points[:,1] = ground_truth_points[:,1] + 25
test1_list = np.array([[-6474.522619870042, 826.0059592951886], [-5717.648251339817, 3074.8272266829454], [-2814.562960434822, 2065.264230491675], [4271.691417490318, 3305.06481536135], [527.9110599415193, 3631.832858597904], [2102.265126128572, 5692.196881649792], [-1534.1904676645468, 5835.7018094593595], [-5687.803032037682, 5741.7436958421395], [5020.6518524696885, 8602.01458239936], [-744.7189096150678, 9001.245344154428]]) #[-1534.1904676645468, 5835.7018094593595]
test2_list = np.array([[-747.6872312211508, 9037.122761380906], [-5689.2105556783745, 5743.164567828242], [-1535.4617557550873, 5840.537492098952], [2103.3044793550694, 5695.011085777173], [528.1987498924411, 3633.8120590658045], [-2841.197323477437, 2009.4769889363376], [-5722.9325842892995, 3077.6690263379205], [4272.482324992167, 3305.6767510798027]]) #[-1535.4617557550873, 5840.537492098952]
test3_list = np.array([[], [], [], [], [], [], [], [], [], [], []])
test4_list = np.array([[], [], [], [], [], [], [], [], [], [], []])
test5_list = np.array([[], [], [], [], [], [], [], [], [], [], []])
test6_list = np.array([[], [], [], [], [], [], [], [], [], [], []])
test7_list = np.array([[], [], [], [], [], [], [], [], [], [], []])
test8_list = np.array([[], [], [], [], [], [], [], [], [], [], []])
test9_list = np.array([[], [], [], [], [], [], [], [], [], [], []])
test10_list = np.array([[], [], [], [], [], [], [], [], [], [], []])
test11_list = np.array([[], [], [], [], [], [], [], [], [], [], []])
test12_list = np.array([[], [], [], [], [], [], [], [], [], [], []])

def rotate_points(ref_point, list_of_points):
    theta =-1*( np.arctan2(ref_point[1], ref_point[0]) - np.deg2rad(90))
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

# Example usage
rotated_point_list1 = rotate_points([-1534.1904676645468, 5835.7018094593595], test1_list)
rotated_point_list2 = rotate_points([-1535.4617557550873, 5840.537492098952], test2_list)

create_point_cloud(rotated_point_list1)
create_point_cloud(rotated_point_list2)

