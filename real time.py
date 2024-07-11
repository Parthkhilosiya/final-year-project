import cv2

def capture_and_save_image(save_path):
    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(1)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 's' to take a screenshot and save it to a file.")
    print("Press 'q' to quit.")

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF
        
        # Check for key presses
        if key == ord('s'):
            # Save the frame to a file
            cv2.imwrite(save_path, frame)
            print(f"Screenshot saved to {save_path}.")
        elif key == ord('q'):
            # Quit the program
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Specify the path where the screenshot should be saved
save_path = 'screenshot.jpg'

# Call the function to capture and save an image from the webcam
capture_and_save_image(save_path)
