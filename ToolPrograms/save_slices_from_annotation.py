import os
import yaml
import cv2

def save_slices_from_ann(supervisely_path, img_path, output_path, num_of_files = -1):
    for filename, image in zip(os.listdir(supervisely_path)[:num_of_files], os.listdir(img_path)[:num_of_files]):
        img = cv2.imread(img_path+"\\"+image)

        i = 0

        if filename.endswith(".json"):
            with open(os.path.join(supervisely_path, filename), 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                cv2.imshow("Show", img)
                cv2.waitKey()
                print()
                print(image)


                for obj in data['objects']:
                    x1, y1, x2, y2 = obj['points']['exterior'][0] + obj['points']['exterior'][1]
                    print(i)

                    output_path_slice = output_path + "\\" + obj['classTitle'] + "\\" + f"slice{i}_{image}"

                    cv2.imwrite(output_path_slice, img[y1:y2, x1:x2])

                    i += 1


supervisely_path = "Images\\OwnData\\DataBatch2\\Labels\\DataBatch2\\ann"
img_path = "Images\\OwnData\\DataBatch2\\Images"
output_path = "Images\\ConeSlices"

save_slices_from_ann(supervisely_path, img_path, output_path)


