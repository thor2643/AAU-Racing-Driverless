from ultralytics import YOLO
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


if __name__ == '__main__':

    for folder in os.listdir("runs\detect"):
        path_to_weights = f"runs\detect\\{folder}\\weights\\best.pt"

        # Load a model
        model = YOLO(path_to_weights)  # load a custom model

        # Validate the model
        metrics = model.val(data="YOLO\config.yaml", save_json = True, split = "val", classes = [0,1,2,3])  # no arguments needed, dataset and settings remembered




