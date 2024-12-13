import os

def validating_model(file_path = r"D:/Image Recognition and Detection/yolov5/val.py",
                     best_trained_model = r"D:/Image Recognition and Detection/yolov5/runs/train/exp/weights/best.pt",
                     data_path = r"D:/Image Recognition and Detection/config/data.yaml",
                     img_size = 640):
    # Wrap paths containing spaces in double quotes
    command = f'python "{file_path}" --weights "{best_trained_model}" --data "{data_path}" --img {img_size}'
    os.system(command)

def testing_model(file_path = r"D:/Image Recognition and Detection/yolov5/detect.py",
                  best_trained_model = r"D:/Image Recognition and Detection/yolov5/runs/train/exp2/weights/best.pt",
                  data_path = r"D:/Image Recognition and Detection/config/data.yaml",
                  img_size = 640,
                  test_path = r"D:/Image Recognition and Detection/dataset/images/test",
                  project_path = r"D:\Image Recognition and Detection\results\test_results",
                  exp_name = "yolov5exp",
                  conf_threshold = 0.45):
    # Wrap paths containing spaces in double quotes
    command = f'python "{file_path}" --weights "{best_trained_model}" --img {img_size} --conf {conf_threshold} --source "{test_path}" --project "{project_path}" --name "{exp_name}"'
    os.system(command)

if __name__ == "__main__":
    testing_model()

