import cv2
import numpy as np
import os

def extract_seed_features(image):
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

def soybean_segmentation(image_path, background_path, output_directory):
    try:
        # Read the input images
        captured_image = cv2.imread(image_path)
        background = cv2.imread(background_path)

        # Check if images are successfully loaded
        if captured_image is None:
            raise ValueError("Unable to read the captured image")
        if background is None:
            raise ValueError("Unable to read the background image")

        # Resize the background image to match the dimensions of the captured image
        background = cv2.resize(background, (captured_image.shape[1], captured_image.shape[0]))

        # Perform background subtraction
        frame_difference = cv2.absdiff(captured_image, background)

        # Convert to grayscale for further processing
        gray_frame_difference = cv2.cvtColor(frame_difference, cv2.COLOR_BGR2GRAY)

        # Apply locally adaptive thresholding
        _, binary_image = cv2.threshold(gray_frame_difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform morphological operations to remove noise and fill holes
        kernel = np.ones((5, 5), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        # Find contours of soybean seeds
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a directory to save output images if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Save original image
        cv2.imwrite(os.path.join(output_directory, 'original_image.jpg'), captured_image)

        # Save binary image
        cv2.imwrite(os.path.join(output_directory, 'binary_image.jpg'), cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR))

        # Initialize a list to store features
        features = []

        # Iterate through each contour and crop the soybean seed
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            soybean_seed = captured_image[y:y+h, x:x+w]

            # Save cropped seed images for feature extraction
            cv2.imwrite(os.path.join(output_directory, f"soybean_seed_{i+1}.jpg"), soybean_seed)

            # Extract features from the soybean seed (example: average color and entropy)
            average_color, entropy = extract_seed_features(soybean_seed)
            color_label = categorize_color(average_color)
            entropy_label = categorize_entropy(entropy)

            # Append features to the list
            features.append((f"soybean_seed_{i+1}.jpg", average_color, entropy, color_label, entropy_label))

            # Draw rectangle around the segmented seed on the original image
            cv2.rectangle(captured_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the results
        cv2.imshow("Original Image", captured_image)
        cv2.imshow("Binary Image", binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Output images saved successfully.")

        return features

    except Exception as e:
        print("Error:", e)
        return None

def feature_extraction_test(image_path, background_path, output_directory):
    # Perform segmentation and feature extraction
    seed_features = soybean_segmentation(image_path, background_path, output_directory)

    # Display or save features
    if seed_features:
        for filename, average_color, entropy, color_label, entropy_label in seed_features:
            print(f"Seed: {filename}, Color: {color_label}, Entropy: {entropy_label}")
    else:
        print("Feature extraction failed.")

# Example usage for testing feature extraction
image_path = 'local.jpg'
background_path = 'output_image.jpg'
output_directory = 'output_images'

feature_extraction_test(image_path, background_path, output_directory)
