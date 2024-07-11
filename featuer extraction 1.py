import cv2
import numpy as np
import os

def extract_seed_features(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate average color
    average_color = np.mean(image, axis=(0, 1))

    # Calculate texture features (example: using Haralick texture features)
    textures = cv2.calcHist([gray], [0], None, [256], [0, 256])
    entropy = -np.sum(textures * np.log2(textures + 1e-10))  # Entropy as a texture feature

    return average_color, entropy

def categorize_color(average_color):
    blue, green, red = average_color
    if red > green and red > blue:
        return "Red"
    elif green > red and green > blue:
        return "Green"
    elif blue > red and blue > green:
        return "Blue"
    else:
        return "Mixed"

def categorize_entropy(entropy):
    if entropy < -1000:
        return "Very High"
    elif entropy < -500:
        return "High"
    elif entropy < 0:
        return "Medium"
    else:
        return "Low"

# Directory containing cropped seed images
seed_directory = 'output_images'

# Extract features from each seed image in the directory
seed_features = []
for filename in os.listdir(seed_directory):
    if filename.startswith("soybean_seed_") and filename.endswith(".jpg"):
        seed_path = os.path.join(seed_directory, filename)
        average_color, entropy = extract_seed_features(seed_path)
        color_label = categorize_color(average_color)
        entropy_label = categorize_entropy(entropy)
        seed_features.append((filename, average_color, entropy, color_label, entropy_label))

# Display or save features
for filename, average_color, entropy, color_label, entropy_label in seed_features:
    print(f"Seed: {filename}, Color: {color_label}, Entropy: {entropy_label}")
