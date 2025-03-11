import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import sys

running = True  # متغیر سراسری برای کنترل اجرای حلقه
stop_requested = False # متغیر سراسری برای شناسایی فشار دادن کلید 'q'

# تابع برای بستن برنامه
def end_action():
    global running
    running = False  # توقف پردازش
    root.quit()  # بستن پنجره tkinter و خروج از برنامه

# تابع برای تنظیم وضعیت کلید 'q'
def key_callback(event):
    global stop_requested
    if event.keysym == 'q':
        stop_requested = True
        end_action()  # بستن برنامه



root = tk.Tk()
root.title("Prespective Crop plaque")
root.geometry("500x500")
root.config(bg="#dda0dd")


# Function to process each image
def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image for KMeans clustering
    X = image_rgb.reshape(-1, 3)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(X)

    # Get the segmented image
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(image_rgb.shape)

    # Convert to grayscale and threshold the image
    gray_image = cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Handle case where no contours are found
    if not contours:
        print("No contours found")
        return

    # Assume the largest contour is the license plate
    plate_contour = max(contours, key=cv2.contourArea)

    # Find the minimum area rectangle around the contour
    rect = cv2.minAreaRect(plate_contour)
    corners = cv2.boxPoints(rect)
    corners = np.int32(corners)

    # Sort the corners to the correct order
    def sort_corners(corners):
        cX = np.mean(corners[:, 0])
        cY = np.mean(corners[:, 1])
        sorted_corners = sorted(corners, key=lambda p: (np.arctan2(p[1] - cY, p[0] - cX)))
        sorted_corners = np.array(sorted_corners)
        top_left = min(sorted_corners, key=lambda x: x[0] + x[1])
        bottom_right = max(sorted_corners, key=lambda x: x[0] + x[1])
        top_right = max(sorted_corners, key=lambda x: x[0] - x[1])
        bottom_left = min(sorted_corners, key=lambda x: x[0] - x[1])
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    sorted_corners = sort_corners(corners)

    # Handle case where not all corners are detected correctly
    if len(sorted_corners) < 4:
        center_x, center_y = np.mean(image.shape[1]), np.mean(image.shape[0])
        sorted_corners = np.array([
            [center_x - 100, center_y - 50],
            [center_x + 100, center_y - 50],
            [center_x + 100, center_y + 50],
            [center_x - 100, center_y + 50]
        ], dtype=np.float32)

    # Print sorted corners
    print("Sorted Corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left):")
    print(sorted_corners)

    # Clone image for displaying draggable points
    image_with_points = image.copy()

    # Variables for dragging points
    global dragging, dragged_point_index
    dragging = False
    dragged_point_index = -1

    def draw_points(img, points):
        """Draw points on the image."""
        for idx, point in enumerate(points):
            cv2.circle(img, tuple(point.astype(int)), 5, (0, 255, 0), -1)

    def mouse_callback(event, x, y, flags, param):
        global dragging, dragged_point_index
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(sorted_corners):
                if abs(point[0] - x) < 10 and abs(point[1] - y) < 10:
                    dragging = True
                    dragged_point_index = i
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging and dragged_point_index != -1:
                sorted_corners[dragged_point_index] = [x, y]
                image_with_points[:] = image.copy()
                draw_points(image_with_points, sorted_corners)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            dragged_point_index = -1

    # Set up the mouse callback function
    cv2.namedWindow('Original Image with Points')
    cv2.setMouseCallback('Original Image with Points', mouse_callback)

    # Draw initial points
    draw_points(image_with_points, sorted_corners)

    # Display the original image with draggable points
    while True:
        cv2.imshow('Original Image with Points', image_with_points)
        key = cv2.waitKey(1)
        if key == 27:  # Press Esc to exit
            break
        elif key == ord(' '):  # Press Space to save and proceed to the next image
            # Calculate width and height based on detected corners
            width = max(np.linalg.norm(sorted_corners[0] - sorted_corners[1]), np.linalg.norm(sorted_corners[2] - sorted_corners[3]))
            height = max(np.linalg.norm(sorted_corners[0] - sorted_corners[3]), np.linalg.norm(sorted_corners[1] - sorted_corners[2]))

            # Define destination points based on calculated width and height
            dst_np = np.array([
                [0, 0],
                [int(width), 0],
                [int(width), int(height)],
                [0, int(height)]
            ], dtype=np.float32)

            # Perform perspective transformation
            M = cv2.getPerspectiveTransform(sorted_corners, dst_np)
            result = cv2.warpPerspective(image, M, dsize=(int(width), int(height)))

            # Resize the output image to the fixed size of 400x80
            final_size = (400, 80)
            result_resized = cv2.resize(result, final_size)

            # Save the transformed image
            cv2.imwrite(output_path, result_resized)
            print(f"Image saved as '{output_path}'")
            break
        elif key == ord('n'):  # Press 'n' to skip the current image (don't save)
            print(f"Skipped image: {image_path}")
            cv2.destroyAllWindows()
            return False  # Return False to indicate image was skipped
        elif key == ord('q'):  # Press 'q' to quit the program
            print("Quitting program...")
            cv2.destroyAllWindows()
            return True  # Return True to indicate the user wants to quit

    # Close all OpenCV windows  
    cv2.destroyAllWindows()
    return False  # Return False to indicate image was processed and saved


# Define input and output directories
input_dir = 'input'
output_dir = 'output'

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
# تعریف دکمه‌های شروع و پایان
def start_action():
    global running, stop_requested
    messagebox.showinfo("start", "با نقاط سبز محل چهارگوشه پلاک رو مشخص کن")
# Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            output_filename = f"CROP_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            # Call process_image and check if 'q' was pressed to quit
            if process_image(image_path, output_path):
                break  # If True is returned, quit the program
            root.update()  # به روز رسانی پنجره tkinter



# دکمه استارت و پایان
label = tk.Label(root, text="Prespective Crop plaque", font=("Arial", 14), bg="#dda0dd", fg="black")
label.pack(pady=20)

label = tk.Label(root, text="language = ENG \n\n Caps Lock = off \n\n exit = q \n\n skip =  n \n\n if its okay = space ", font=("Arial", 14), bg="#dda0dd", fg="black")
label.pack(pady=20)

start_button = tk.Button(root, text="Start", font=("Arial", 12, "bold"), command=start_action, bg="#ffa500", fg="black")
start_button.pack(pady=10)

end_button = tk.Button(root, text="End", font=("Arial", 12, "bold"), command=end_action, bg="#ffa500", fg="black")
end_button.pack(pady=10)

root.mainloop()