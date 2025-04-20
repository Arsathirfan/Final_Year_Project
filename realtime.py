import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("modelnew.h5")

# Define image size (should match what was used during training)
IMG_SIZE = 128

# Define labels
CLASSES = ["NonViolence", "Violence"]

# Start video capture (0 for default webcam, or provide video file path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))  # Resize to match model input
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame_normalized = frame_rgb / 255.0  # Normalize pixel values

    # Expand dimensions to match model input shape (batch_size, IMG_SIZE, IMG_SIZE, 3)
    frame_expanded = np.expand_dims(frame_normalized, axis=0)

    # Make prediction
    prediction = model.predict(frame_expanded)[0][0]

    # Determine label and confidence score
    label = CLASSES[1] if prediction > 0.5 else CLASSES[0]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Display label on frame
    text = f"{label}: {confidence:.2f}"
    color = (0, 0, 255) if label == "Violence" else (0, 255, 0)  # Red for Violence, Green for NonViolence
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show video feed
    cv2.imshow("Fight Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()