import cv2
import numpy as np

def soybean_segmentation(image_path, background_path):
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

        # Iterate through each contour and crop the soybean seed
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            soybean_seed = captured_image[y:y+h, x:x+w]

            # Get the size of the segmented seed
            seed_size = f"Seed {i+1} - Width: {w}px, Height: {h}px"

            # Draw rectangle around the segmented seed
            cv2.rectangle(captured_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display size information on image window
            cv2.putText(captured_image, seed_size, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print size information in the terminal
            print(seed_size)

        # Display the results
        cv2.imshow("Original Image", captured_image)
        cv2.imshow("Binary Image", binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

# Example usage
image_path = 'testing 2.jpeg'
background_path = 'output_image.jpg'
soybean_segmentation(image_path, background_path)
