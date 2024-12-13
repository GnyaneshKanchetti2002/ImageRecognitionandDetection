import os

def train_yolov5(train_file = r"D:\Image Recognition and Detection\yolov5\train.py",
                 data_yaml=r"D:\Image Recognition and Detection\config\data.yaml",
                 model=r"D:\Image Recognition and Detection\yolov5\yolov5s.pt",
                 epochs=100,
                 batch_size=1,
                 img_size=640):
    command = f'python "{train_file}" --img "{img_size}" --batch {batch_size} --epochs {epochs} --data "{data_yaml}" --weights "{model}" --cache'
    os.system(command)

def retrain_yolov5(train_file = r"D:\Image Recognition and Detection\yolov5\train.py",
                   data_yaml = r"D:\Image Recognition and Detection\config\data.yaml",
                   model = r"D:/Image Recognition and Detection/yolov5/runs/train/exp/weights/best.pt",
                   epochs = 20,
                   batch_size = 1,
                   img_size = 640,
                   hyp = r"D:\Image Recognition and Detection\yolov5\data\hyps\hyp.scratch-med.yaml"):
    command = f'python "{train_file}" --img {img_size} --batch {batch_size} --epochs {epochs} --data "{data_yaml}" --weights "{model}" --hyp "{hyp}" --label-smoothing 0.01'
    os.system(command)

if __name__ == "__main__":
    retrain_yolov5()