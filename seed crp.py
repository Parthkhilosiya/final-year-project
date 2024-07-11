import cv2
import numpy as np
import os

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

        # Apply Gaussian blur to smooth the image
        frame_difference_blurred = cv2.GaussianBlur(frame_difference, (5, 5), 0)

        # Convert to grayscale for further processing
        gray_frame_difference = cv2.cvtColor(frame_difference_blurred, cv2.COLOR_BGR2GRAY)

        # Apply thresholding using Otsu's method
        _, binary_image = cv2.threshold(gray_frame_difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform morphological operations to remove noise and refine the binary image
        kernel = np.ones((5, 5), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # Find contours of soybean seeds
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a directory to save output images if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Save original image
        cv2.imwrite(os.path.join(output_directory, 'original_image.jpg'), captured_image)

        # Save binary image
        cv2.imwrite(os.path.join(output_directory, 'binary_image.jpg'), binary_image)

        # Iterate through each contour and crop the soybean seed
        for i, contour in enumerate(contours):
            # Ignore small or irrelevant contours (adjust the area threshold if necessary)
            if cv2.contourArea(contour) < 50:
                continue
            
            # Compute the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Add padding around the bounding box for better cropping (adjust padding if necessary)
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w + 2 * padding, captured_image.shape[1] - x)
            h = min(h + 2 * padding, captured_image.shape[0] - y)

            # Crop the soybean seed image
            soybean_seed = captured_image[y:y + h, x:x + w]

            # Save cropped seed images for feature extraction
            cv2.imwrite(os.path.join(output_directory, f"soybean_seed_{i + 1}.jpg"), soybean_seed)

        print("Output images saved successfully.")

    except Exception as e:
        print("Error:", e)

# Example usage
image_path = 'local 2.jpg'
background_path = 'output_image.jpg'
output_directory = 'output_images'
soybean_segmentation(image_path, background_path, output_directory)
