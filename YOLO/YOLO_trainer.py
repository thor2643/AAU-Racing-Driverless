from ultralytics import YOLO
import yaml



default_params = {"epochs": 100,
                  "batch": 16,
                  "optimizer": 'auto',
                  "freeze": None,
                  "lr0": 0.01,
                  "lrf": 0.01,
                  "momentum": 0.937,
                  "weight_decay": 0.0005,
                  }

with open("YOLO\hyperparam_tuning\\optimizer.json", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)


#Very important that the code is inside __main__
#Causes errors otherwise (something to do with enabling gpu)
if __name__ == '__main__':
    for config in data["trainings"]:
        params = default_params.copy()

        for param in config:
            params[param] = config[param]


        model = YOLO("YOLO\\yolov8n.pt")

        results = model.train(data="YOLO\config.yaml", 
                              epochs = params["epochs"],
                              batch = params["batch"],
                              optimizer = params["optimizer"],
                              freeze = params["freeze"],
                              lr0 = params["lr0"],
                              lrf = params["lrf"],
                              momentum = params["momentum"],
                              weight_decay = params["weight_decay"])