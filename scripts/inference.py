import os
import cv2
import numpy as np
from openvino.runtime import Core

def load_model(model_xml_path, device="GPU"):
    core = Core()
    model = core.read_model(model_xml_path)
    compiled_model = core.compile_model(model, device)
    return compiled_model

def preprocess_image(image, input_size):
    image_resized = cv2.resize(image, input_size)
    image_transposed = image_resized.transpose(2, 0, 1)  # HWC to CHW
    image_expanded = np.expand_dims(image_transposed, axis=0)  # Add batch dimension
    return image_expanded


def postprocess_result(output, image, conf_threshold=0.1):  # Lowered threshold for testing
    pass
    return boxes, confidences, class_ids


def draw_boxes(image, boxes):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

def process_images(input_folder, output_folder, model, input_shape):
    for img_file in os.listdir(input_folder):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_folder, img_file)
            image = cv2.imread(img_path)

            # Preprocess
            input_image = preprocess_image(image, input_shape)

            # Run inference
            output = model(input_image)[model.outputs[0]]

            # Postprocess
            boxes, confidences, class_ids = postprocess_result(output, image)

            # Draw bounding boxes on the original image
            draw_boxes(image, boxes)

            # Save the output image to the output folder
            output_img_path = os.path.join(output_folder, img_file)
            cv2.imwrite(output_img_path, image)
            print(f"Processed and saved: {output_img_path}")

    return None