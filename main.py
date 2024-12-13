import os
from scripts import inference

def main():
    # Paths to model files
    model_xml = r"D:\Image Recognition and Detection\models\best.xml"
    model_bin = r"D:\Image Recognition and Detection\models\best.bin"

    # Load the model
    compiled_model = inference.load_model(model_xml)

    # Input and output folders
    input_folder = r"D:\Image Recognition and Detection\dataset\images\test"
    output_folder = r"D:\Image Recognition and Detection\results\test_results\OpenVino_yolov5"
    os.makedirs(output_folder, exist_ok=True)

    # Define input shape (should match the model's input size)
    input_shape = (640, 640)

    # Process all images in the input folder and save results in the output folder
    inference.process_images(input_folder, output_folder, compiled_model, input_shape)

if __name__ == "__main__":
    main()
