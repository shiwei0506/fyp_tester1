import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')  # Use yolov8n.pt (nano version), yolov8s.pt (small), or another version depending on your needs

# Function to detect items using YOLOv8
def detect_objects_yolov8(image):
    # YOLOv8 expects the image to be in RGB format, OpenCV reads in BGR, so we need to convert it
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run the YOLOv8 model on the image
    results = model(image_rgb)

    # Get the detected objects (bounding boxes and class IDs)
    boxes = results.xywh[0].numpy()  # Get bounding boxes (xywh)
    item_count = len(boxes)  # Count the number of detected items

    return item_count, boxes

# Streamlit app to capture and process image
st.title("Real-Time Inventory Monitor")

img_file_buffer = st.camera_input("Capture stock items image.")

if img_file_buffer is not None:
    # Save the captured image to a file
    with open('captured_image.jpg', 'wb') as file:
        file.write(img_file_buffer.getbuffer())

    # Load the image using OpenCV
    img = cv2.imread('captured_image.jpg')

    # Run object detection with YOLOv8
    item_count, boxes = detect_objects_yolov8(img)

    # Display the detected item count
    st.write(f"Detected {item_count} items.")

    # Optional: Draw bounding boxes around detected items
    for box in boxes:
        x, y, w, h = map(int, box)
        cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)

    # Display the processed image with bounding boxes
    st.image(img, channels="BGR", use_column_width=True)

    # Add volume recommendation based on item count
    if item_count < 5:
        st.write("Small volume of stock.")
    elif 5 <= item_count < 15:
        st.write("Medium volume of stock.")
    else:
        st.write("Large volume of stock.")
