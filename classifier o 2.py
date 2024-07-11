import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

# Define the path to your dataset
dataset_root = 'C:/Users/parth khiloshiya/Desktop/final year project/Soybean Seeds'

# Check if the dataset path exists
if not os.path.exists(dataset_root):
    raise FileNotFoundError(f"The dataset path '{dataset_root}' does not exist.")

# Get the list of classes (subfolders) in the dataset
classes = os.listdir(dataset_root)

data = []
labels = []

# Load and preprocess images from each class
for class_name in classes:
    class_folder = os.path.join(dataset_root, class_name)

    for filename in os.listdir(class_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(class_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue
            image = cv2.resize(image, (128, 128))  # Resize images to a consistent size
            data.append(image)
            labels.append(class_name)

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save the label encoder classes
np.save('label_encoder_classes.npy', label_encoder.classes_)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=len(classes))

# Apply data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the CNN model with dropout and batch normalization
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Get the shape of the input images
input_shape = (128, 128, 3)

# Create the CNN model
model = create_model(input_shape, len(classes))

# Set early stopping and learning rate scheduler callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
learning_rate_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train_one_hot, batch_size=32),
    epochs=25,
    validation_data=(X_test, y_test_one_hot),
    callbacks=[early_stopping, learning_rate_scheduler]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Save the model for future use
model.save('improved_soybean_classifier_model.h5')
