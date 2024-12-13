import os

def train_yolov8(data_yaml = r"D:\Image Recognition and Detection\config\data.yaml",
                 model_path = r"D:\Image Recognition and Detection\yolov8\yolov8s.pt",
                 project_path = r"D:\Image Recognition and Detection\yolov8",
                 name = "yolov8_custom_training"):
    command = f'yolo task = detect mode = train model = "{model_path}" data = "{data_yaml}" epochs = 100 imgsz = 640 batch = 1 project = "{project_path}" name = "{name}"'
    os.system(command)

def retrain_yolov8(model_path = r"D:\Image Recognition and Detection\yolov8\yolov8_custom_training\weights\best.pt",
                   data = r"D:\Image Recognition and Detection\config\data.yaml",
                   project_path=r"D:\Image Recognition and Detection\yolov8",
                   name="yolov8_custom_training"):
    command = f'yolo task = detect mode = train model = "{model_path}" data = "{data}" epochs = 50 imgsz = 640 batch = 1 patience = 10 label_smoothing = 0.1 lr0 = 0.001 weight_decay = 0.001 project = "{project_path}" name = "{name}"'
    os.system(command)

if __name__ == "__main__":
    retrain_yolov8()
