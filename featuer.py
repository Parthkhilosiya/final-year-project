import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_seed_features(image_path):
    # Read the image from the specified path
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found or unable to open the image file")

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram of the Hue channel
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])

    # Plot the histogram of the Hue channel
    plt.figure()
    plt.title("Histogram of Hue Channel")
    plt.xlabel("Hue Value")
    plt.ylabel("Frequency")
    plt.plot(hist_hue)
    plt.xlim([0, 256])
    plt.show()

    # Calculate average color in BGR format
    average_color = np.mean(image, axis=(0, 1))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram of the grayscale image
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Calculate entropy of the grayscale histogram
    hist_gray_normalized = hist_gray / np.sum(hist_gray)
    entropy = -np.sum(hist_gray_normalized * np.log2(hist_gray_normalized + 1e-10))

    return average_color, entropy

def categorize_color(average_color):
    # Convert average color from BGR to HSV to categorize it
    color_hsv = cv2.cvtColor(np.uint8([[average_color]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # Hue thresholds for color categorization
    if color_hsv[0] < 30 or color_hsv[0] > 330:
        return "Red"
    elif color_hsv[0] >= 30 and color_hsv[0] <= 85:
        return "Yellow"
    elif color_hsv[0] > 85 and color_hsv[0] <= 150:
        return "Green"
    elif color_hsv[0] > 150 and color_hsv[0] <= 240:
        return "Blue"
    else:
        return "Mixed"

def categorize_entropy(entropy):
    # Entropy thresholds for categorization (adjust these as needed)
    if entropy < 3:
        return "Low"
    elif entropy < 5:
        return "Medium"
    else:
        return "High"

# Specify the path to the input image
input_image_path = 'testing 2.jpeg'  # Update with the correct path to the input image

# Extract features from the input image
average_color, entropy = extract_seed_features(input_image_path)

# Categorize the color and entropy of the input image
color_label = categorize_color(average_color)
entropy_label = categorize_entropy(entropy)

# Display the classification results
print(f"Average Color (BGR): {average_color}")
print(f"Entropy: {entropy:.2f}")
print(f"Color Label: {color_label}")
print(f"Entropy Label: {entropy_label}")
