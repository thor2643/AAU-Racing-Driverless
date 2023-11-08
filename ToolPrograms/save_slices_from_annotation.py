import os
import yaml
import cv2

def save_slices_from_ann(supervisely_path, img_path, output_path, num_of_files = -1):
    for filename, image in zip(os.listdir(supervisely_path)[:num_of_files], os.listdir(img_path)[:num_of_files]):
        img = cv2.imread(img_path+"\\"+image)
        
        if filename.endswith(".json"):
            with open(os.path.join(supervisely_path, filename), 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)


                for obj in data['objects']:
                    x1, y1, x2, y2 = obj['points']['exterior'][0] + obj['points']['exterior'][1]

                    output_path = output_path + "\\" + obj['classTitle'] + "\\"

                    cv2.imwrite
                    


