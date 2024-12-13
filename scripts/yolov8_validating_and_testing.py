import os

def validating_model(model_path = r"D:\Image Recognition and Detection\yolov8\yolov8_custom_training\weights\best.pt",
                     data = r"D:\Image Recognition and Detection\config\data.yaml",
                     project_path=r"D:\Image Recognition and Detection\yolov8",
                     name="yolov8_custom_validating"):
    command = f'yolo task=detect mode=val model="{model_path}" data="{data}" project = "{project_path}" name = "{name}"'
    os.system(command)

def testing_model(model_path = r"D:\Image Recognition and Detection\yolov8\yolov8_custom_training2\weights\best.pt",
                  test_path = r"D:/Image Recognition and Detection/dataset/images/test",
                  project_path=r"D:\Image Recognition and Detection\results\test_results",
                  exp_name="yolov8exp"):
    command = f'yolo task = detect mode = predict model = "{model_path}" source = "{test_path}" project = "{project_path}" name = "{exp_name}"'
    os.system(command)

if __name__ == '__main__':
    testing_model()