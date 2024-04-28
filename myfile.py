import cv2
import pytesseract
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import easyocr
import re
import json
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO

# Confidence threshold for object detection
CONFIDENCE_THRESHOLD = 0.3
logomod = "Logo model/best.pt"
carcol = "Car color model/best.pt"
licol = "License Plate color model/best.pt"

# Initialize dictionary to store results
results_dict = {"objects_detected": [], "sticker_count": 0, "flag_count": 0}


def process_image(image):
  """
  Analyzes an image and returns a dictionary containing the results.

  Args:
      image_path: Path to the image file.

  Returns:
      A dictionary containing the analysis results.
  """
  # Load the image
 image = cv2.imdecode(np.fromfile(image.file, dtype=np.uint8), cv2.IMREAD_COLOR)  


  # Object detection using YOLO models
  model = YOLO('Detection Model/best.pt')
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


def crop_and_display_license_plate(image, license_plate_box):
  """
  Extracts and analyzes the license plate information.

  Args:
      image: The image data.
      license_plate_box: Bounding box coordinates of the license plate.
  """
  global results_dict
  x1, y1, x2, y2 = license_plate_box
  license_plate_image = image[int(y1):int(y2), int(x1):int(x2)]

  # License plate color detection using YOLO model
  model = YOLO(licol)
  results = model.predict(license_plate_image, conf=0.5)
  top1_confidence = results[0].probs.top1conf
  top1_class_idx = results[0].probs.top1
  best_classification = results[0].names[top1_class_idx]

  # License plate OCR with Tesseract
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
def crop_car(image, car_box):
  """
  Extracts and analyzes the car information (color in this case).

  Args:
      image: The image data.
      car_box: Bounding box coordinates of the car.
  """
  global results_dict
  x1, y1, x2, y2 = car_box
  car_image = image[int(y1):int(y2), int(x1):int(x2)]

  # Car color detection using YOLO model
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


def crop_logo(image, logo_box):
  """
  Extracts and analyzes the logo information (manufacturer in this case).

  Args:
      image: The image data.
      logo_box: Bounding box coordinates of the logo.
  """
  global results_dict
  x1, y1, x2, y2 = logo_box
  logo_image = image[int(y1):int(y2), int(x1):int(x2)]

  # Logo detection using YOLO model
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


# Function to process OCR text and clean it
def clean_text(text):
  """
  Cleans the OCR text by removing non-alphanumeric characters.

  Args:
      text: The OCR text to be cleaned.

  Returns:
      The cleaned text.
  """
  cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)
  return cleaned_text


# Function to process OCR text for license plate
def process_ocr_text(ocr_text):
  """
  Processes the OCR text for license plate by converting it to uppercase 
  and removing non-alphanumeric characters.

  Args:
      ocr_text: The OCR text from the license plate.

  Returns:
      The processed license plate text.
  """
  processed_text = clean_text(ocr_text.upper())
  return processed_text


# Example usage (assuming you have a function to handle image upload)
def analyze_image(uploaded_image):
  """
  Analyzes the uploaded image and returns the results as a dictionary.

  Args:
      uploaded_image: The uploaded image data.

  Returns:
      A dictionary containing the analysis results.
  """
  image_path = save_uploaded_image(uploaded_image)  # Replace with your logic to save the image
  results = process_image(image_path)
  return results




