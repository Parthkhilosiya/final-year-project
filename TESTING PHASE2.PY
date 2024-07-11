
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
from tensorflow.keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Filter out the deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")


# Load the trained model
trained_model = load_model('C:/Users/parth khiloshiya/Desktop/final year project/improved_soybean_classifier_model.h5')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('C:/Users/parth khiloshiya/Desktop/final year project/label_encoder_classes.npy')

# Function to classify a new image
def classify_image(image_path, label_encoder):
    # Read and preprocess the new image
    new_image = cv2.imread(image_path)
    new_image = cv2.resize(new_image, (128, 128)) / 255.0
    new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension

    # Make a prediction using the trained model
    prediction = trained_model.predict(new_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # Get the predicted class label
    predicted_class = label_encoder.classes_[predicted_class_index]

    return predicted_class

# Path to the folder containing the images you want to classify
output_image_folder = 'C:/Users/parth khiloshiya/Desktop/final year project/output_images'

# List of image files to classify
image_files = os.listdir(output_image_folder)

# Classify each image in the folder
for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(output_image_folder, image_file)
    
    # Classify the image and print the result
    predicted_class = classify_image(image_path, label_encoder)
    print(f"The predicted class for the image '{image_file}' is: {predicted_class}")

