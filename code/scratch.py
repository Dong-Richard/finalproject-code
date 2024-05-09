import os
from xml.etree import ElementTree
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt

# Function to parse XML files and extract bounding box information
def parse_annotation(xml_path):
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    path = root.find('path').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymin = int(obj.find('bndbox/ymin').text)
        ymax = int(obj.find('bndbox/ymax').text)
        
        bbox = {
            'class': class_name,
            'bbox': (xmin, ymin, xmax, ymax),
            'image_width': width,
            'image_height': height
        }
        
        objects.append(bbox)
    
    return {
        'filename': filename,
        'path': path,
        'objects': objects
    }

# Load and preprocess an image (resize, normalize)
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image

# Normalize bounding boxes to [0, 1]
def normalize_bounding_box(bbox, img_width, img_height):
    xmin, ymin, xmax, ymax = bbox
    xmin_norm = xmin / img_width
    ymin_norm = ymin / img_height
    xmax_norm = xmax / img_width
    ymax_norm = ymax / img_height
    return (xmin_norm, ymin_norm, xmax_norm, ymax_norm)

# Load dataset from directories
def load_dataset(images_dir, annotations_dir):
    dataset = []
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    
    for annotation_file in annotation_files:
        xml_path = os.path.join(annotations_dir, annotation_file)
        parsed_data = parse_annotation(xml_path)
        dataset.append(parsed_data)
    
    return dataset

# Build the CNN model for bounding box regression
def build_model(input_shape=(224, 224, 3), num_bboxes=5):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        # Output: 4 coordinates for each bounding box (xmin, ymin, xmax, ymax)
        layers.Dense(4 * num_bboxes)  # Output flattened bounding box coordinates
    ])
    
    # Loss function for bounding box regression (Mean Squared Error)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# Define paths to images and annotations
images_dir = "C:/Users/jordan/Documents/cs1430/final project/finalproject-code/data/test"
annotations_dir = images_dir  # Assuming annotations are in the same directory

# Load dataset
dataset = load_dataset(images_dir, annotations_dir)

# Shuffle the dataset and split into training and validation sets
np.random.shuffle(dataset)
train_dataset = dataset[:int(0.8 * len(dataset))]
val_dataset = dataset[int(0.8 * len(dataset)):]

# Prepare training data
train_images = []
train_bboxes = []

train_images = []
train_bboxes = []

# Build the training dataset
for item in train_dataset:
    image_path = os.path.join(images_dir, item['filename'])
    train_images.append(load_and_preprocess_image(image_path))
    
    bboxes = []
    for obj in item['objects']:
        bbox = obj['bbox']
        img_width = obj['image_width']
        img_height = obj['image_height']
        
        normalized_bbox = normalize_bounding_box(bbox, img_width, img_height)
        bboxes.append(normalized_bbox)
    
    # Ensure a fixed number of bounding boxes by padding with zeros
    while len(bboxes) < 5:
        bboxes.append((0, 0, 0, 0))
    
    # Flatten to create a single array for each set of bounding boxes
    flattened_bboxes = np.array(bboxes).flatten()[:20]
    
    # Append to the list
    # print(len(flattened_bboxes))
    train_bboxes.append(flattened_bboxes)

# Convert to numpy arrays with the correct shape
train_images = np.array(train_images)
train_bboxes = np.array(train_bboxes)
print(train_bboxes.shape)

# Ensure the shape is correct (number of samples, 20)
assert train_bboxes.shape == (len(train_images), 20), "Incorrect shape for train_bboxes"


# Prepare validation data similarly
val_images = []
val_bboxes = []

for item in val_dataset:
    image_path = os.path.join(images_dir, item['filename'])
    processed_image = load_and_preprocess_image(image_path)
    val_images.append(processed_image)
    
    bboxes = []
    for obj in item['objects']:
        bbox = obj['bbox']
        img_width = obj['image_width']
        img_height = obj['image_height']
        
        normalized_bbox = normalize_bounding_box(bbox, img_width, img_height)
        bboxes.append(normalized_bbox)
    
    while len(bboxes) < 5:
        bboxes.append((0, 0, 0, 0))
    
    val_bboxes.append(np.array(bboxes).flatten())

val_images = np.array(val_images)
val_bboxes = np.array(val_bboxes)

# Check shapes and data types for consistency
print("Training Images Shape:", train_images.shape)  # (num_samples, 224, 224, 3)
print("Training Bounding Boxes Shape:", train_bboxes.shape)  # (num_samples, 20)
# print(train_bboxes[0].shape)
# print(train_bboxes[0].flatten().reshape((train_bboxes.shape[0], -1)).shape)
print("Validation Images Shape:", val_images.shape)  # Should be similar to training images
print("Validation Bounding Boxes Shape:", val_bboxes.shape)  # (num_samples, 20)

print("Are all training bounding boxes numeric?", all(isinstance(item, np.ndarray) for item in train_bboxes))

# Build and train the model
num_bboxes = 5  # Assuming a maximum of 5 bounding boxes per image
model = build_model(num_bboxes=num_bboxes)

model.fit(
    train_images, 
    train_bboxes, 
    epochs=10, 
    validation_data=(val_images, val_bboxes)
)

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, bboxes):
    h, w, _ = image.shape
    for bbox in bboxes:
        xmin = int(bbox[0] * w)
        ymin = int(bbox[1] * h)
        xmax = int(bbox[2] * w)
        ymax = int(bbox[3] * h)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Red
    
    return image

# Predict bounding boxes on validation data
predicted_bboxes = model.predict(val_images)

# Display some validation images with predicted bounding boxes
num_samples = 5  # Display first 5 samples
for i in range(num_samples):
    image = (val_images[i] * 255).astype(np.uint8)  # De-normalize to [0, 255]
    predicted = predicted_bboxes[i].reshape((5, 4))
    
    image_with_pred = draw_bounding_boxes(image.copy(), predicted)
    
    plt.figure(figsize=(8, 6))
    plt.title("Predicted Bounding Boxes")
    plt.imshow(image_with_pred)
    plt.axis('off')
    plt.show()
