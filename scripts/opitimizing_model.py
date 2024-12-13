import os
import sys

def convert_to_onnx(file_path = r"D:\Image Recognition and Detection\yolov5\export.py",
                    best_trained_model = r"D:\Image Recognition and Detection\yolov5\runs\train\exp2\weights\best.pt",
                    img_size = 640,
                    batch_size = 1):
    command = f'python "{file_path}" --weights "{best_trained_model}" --img {img_size} --batch {batch_size} --device cpu --include onnx'
    os.system(command)

def convert_onnx_to_openvino(model_input = r"D:\Image Recognition and Detection\yolov5\runs\train\exp2\weights\best.onnx",
                             model_output = r"D:\Image Recognition and Detection\models",
                             input_shape = [1, 3, 640, 640]):
    input_shape_str = str(input_shape).replace(' ', '')
    sys.path.append(r"D:\openvino\tools\mo\openvino")
    setup_env = r'"D:\openvino\scripts\setupvars\setupvars.bat"'
    mo_path = r'"D:\openvino\tools\mo\openvino\tools\mo\mo.py"'
    command = f'cmd /c {setup_env} && python {mo_path} --input_model "{model_input}" --output_dir "{model_output}" --input_shape "{input_shape_str}"'
    os.system(command)

def convert_onnx_to_IR():
    convert_model(r"D:\Image Recognition and Detection\yolov5\runs\train\exp2\weights\best.onnx",
                  output_dir = r"D:\Image Recognition and Detection\models")

if __name__ == "__main__":
    convert_onnx_to_openvino()
