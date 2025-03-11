import os
from ultralytics import YOLO
import cv2  # OpenCV for image processing

# Load a pretrained YOLOv8 model
model = YOLO("best.pt")

# Define the input and output directories
input_folder = "images"
output_folder = "input"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the padding (in pixels)
padding = 15

# Loop through all images in the input folder
for image_name in os.listdir(input_folder):
    # Create the full path to the image
    image_path = os.path.join(input_folder, image_name)
    
    # Predict on the image
    detection_output = model.predict(source=image_path, conf=0.25)
    
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_name}")
        continue
    image_height, image_width = image.shape[:2]

    # Loop through all the detections for the current image
    for result in detection_output:
        boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates (x1, y1, x2, y2)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Convert to integer

            # Add padding to each side of the bounding box
            x1 = max(0, x1 - padding)  # Ensure x1 is not less than 0
            y1 = max(0, y1 - padding)  # Ensure y1 is not less than 0
            x2 = min(image_width, x2 + padding)  # Ensure x2 is not more than the image width
            y2 = min(image_height, y2 + padding)  # Ensure y2 is not more than the image height

            # Crop the detected license plate area with padding
            cropped_plate = image[y1:y2, x1:x2]

            # Save the cropped image to the output folder
            output_image_name = f"{os.path.splitext(image_name)[0]}_plate_{i}.jpg"
            output_path = os.path.join(output_folder, output_image_name)
            cv2.imwrite(output_path, cropped_plate)
            print(f"Saved cropped plate to: {output_path}")

# Close any OpenCV windows if opened
cv2.destroyAllWindows()
