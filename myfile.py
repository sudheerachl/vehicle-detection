
import cv2
import pytesseract
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import KMeans
import easyocr
import re
import json

# Confidence threshold for object detection
CONFIDENCE_THRESHOLD = 0.3
logomod ="Logo model/best.pt"
carcol =  "Car color model/best.pt"
licol = "License Plate color model/best.pt"

# Initialize dictionary to store results
results_dict = {"objects_detected": [], "sticker_count": 0, "flag_count": 0}
def browse_image():
    global results_dict
    # Update the image label with the selected image path
    image_label.config(text=filename)
    # Display the selected image in the Tkinter window
    display_image_info(filename)
    # Convert results_dict to JSON format and print
    json_output = json.dumps(results_dict, indent=2)
    formatted_json_output = format_json_output(json_output)
    print(formatted_json_output)

def format_json_output(json_output):
    lines = json_output.splitlines()
    start_index = 0
    end_index = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("{"):
            start_index = i
        if line.startswith("}"):
            end_index = i + 1
            break
    return "\n".join(lines[start_index:end_index])

# Function to process OCR text and clean it
def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return cleaned_text

# Function to process OCR text for license plate
def process_ocr_text(ocr_text):
    processed_text = clean_text(ocr_text.upper())
    return processed_text

# Function to extract and display license plate information
def crop_and_display_license_plate(image, license_plate_box):
    global results_dict
    x1, y1, x2, y2 = license_plate_box
    license_plate_image = image[int(y1):int(y2), int(x1):int(x2)]
    model = YOLO(licol)
    results = model.predict(license_plate_image, conf=0.5)
    top1_confidence = results[0].probs.top1conf
    top1_class_idx = results[0].probs.top1
    best_classification = results[0].names[top1_class_idx]
    tesseract_text = pytesseract.image_to_string(license_plate_image, config='--psm 7')
    tesseract_text = clean_text(tesseract_text)
    ocr_text = process_ocr_text(tesseract_text)

    # Update results_dict with license plate information
    results_dict["objects_detected"].append({
        "class_name": "license-plate",
        "bounding_box": [x1, y1, x2, y2],
        "text": ocr_text,
        "color": best_classification
    })

# Function to extract and display car color information
def crop_car(image, car_box):
    global results_dict
    x1, y1, x2, y2 = car_box
    car_image = image[int(y1):int(y2), int(x1):int(x2)]
    model = YOLO(carcol)
    results = model.predict(car_image, conf=0.5)
    
    if results is not None and len(results) > 0:
        if isinstance(results, list):
            results = results[0]
            names_dict = results.names
            boxes = results.boxes.xyxy.tolist()
            class_labels = results.boxes.cls.tolist()

            for i, box in enumerate(boxes):
                class_idx = class_labels[i]
                class_name = results.names[class_idx]

            # Update results_dict with car color information
            results_dict["objects_detected"].append({
                "class_name": "car",
                "bounding_box": [x1, y1, x2, y2],
                "color": class_name
            })
    else:
        # No results for car color detection
        results_dict["objects_detected"].append({
            "class_name": "car",
            "color": "unknown"
        })


# Function to extract and display logo information
def crop_logo(image, logo_box):
    global results_dict
    x1, y1, x2, y2 = logo_box
    logo_image = image[int(y1):int(y2), int(x1):int(x2)]
    model = YOLO(logomod)
    results = model.predict(logo_image, conf=0.5)
    top1_confidence = results[0].probs.top1conf
    top1_class_idx = results[0].probs.top1
    best_classification = results[0].names[top1_class_idx]

    # Update results_dict with logo information
    results_dict["objects_detected"].append({
        "class_name": "logo",
        "bounding_box": [x1, y1, x2, y2],
        "manufacturer": best_classification
    })

def display_image_info(filename):
    global results_dict
    # Load and preprocess the image
    image = cv2.imread(filename)
    model = YOLO('Car color model/best.pt')
    results = model(image, conf=CONFIDENCE_THRESHOLD)
    if isinstance(results, list):
        results = results[0]

    names_dict = results.names
    boxes = results.boxes.xyxy.tolist()
    class_labels = results.boxes.cls.tolist()

    sticker_count = 0
    flag_count = 0

    for i, box in enumerate(boxes):
        class_idx = class_labels[i]
        class_name = results.names[class_idx]

        if class_name == "sticker":
            sticker_count += 1
        elif class_name == "flag":
            flag_count += 1

        if class_name == "license-plate":
            crop_and_display_license_plate(image, box)

        if class_name == "car":
            crop_car(image, box)

        if class_name == "logo":
            crop_logo(image, box)

    # Update results_dict with sticker and flag counts
    results_dict["sticker_count"] = sticker_count
    results_dict["flag_count"] = flag_count

    # Check if logo or car color was detected
    if not any(item["class_name"] == "logo" for item in results_dict["objects_detected"]):
        results_dict["objects_detected"].append({
            "class_name": "logo",
            "manufacturer": "unknown"
        })

    if not any(item["class_name"] == "car" for item in results_dict["objects_detected"]):
        results_dict["objects_detected"].append({
            "class_name": "car",
            "color": "unknown"
        })
    return results_dict

# Create a Tkinter window
"""root = tk.Tk()
root.title("Car Attribute Extraction")

# Create a frame to organize widgets
frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

# Create a label to display instructions
label_instructions = tk.Label(frame, text="Select an image to analyze:")
label_instructions.pack()

# Create a label to display the selected image path
image_label = tk.Label(frame, text="No image selected", font=('Arial', 12), fg='blue')
image_label.pack(pady=10)

# Create a button for browsing images
browse_button = tk.Button(frame, text="Browse Image", command=browse_image, bg='green', fg='white')
browse_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()"""
