import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.saving import custom_object_scope
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from create_model import CNN_BiLSTM_Model  # Import your custom model class from create_model.py


# Function to load sequences from the validate directory
def load_sequences_from_directory(directory):
    sequences = []
    labels = []

    # Walk through the directory and load sequences
    for label, subdir in enumerate(['fight', 'no_fight']):  # Label fight=1, non_fight=0
        subdir_path = os.path.join(directory, subdir)

        for seq_file in os.listdir(subdir_path):
            if seq_file.endswith(".npy"):
                file_path = os.path.join(subdir_path, seq_file)
                sequence = np.load(file_path)  # Load the sequence

                sequences.append(sequence)
                labels.append(label)  # Assign the label (1 for fight, 0 for non-fight)

    return np.array(sequences), np.array(labels)


# Function to create a tf.data.Dataset
def create_tf_dataset(sequences, labels, batch_size=32):
    # Convert the sequences and labels to TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

    return dataset


# Load validation data
val_sequences, val_labels = load_sequences_from_directory("D:\\project_folder\\final_dataset\\test")

# Create the validation dataset
val_dataset = create_tf_dataset(val_sequences, val_labels)

# Load the saved model inside custom_object_scope to handle the custom layer
with custom_object_scope({'CNN_BiLSTM_Model': CNN_BiLSTM_Model}):  # Register the custom model class
    model = tf.keras.models.load_model('cnn_bilstm_fight_detection_model.h5')

# Ensure we're not retraining, just evaluating!
# Get predictions from the model without retraining
y_true = val_labels  # Ground truth labels
y_pred = []

# Iterate over the validation dataset to get predictions
for sequences, _ in val_dataset:
    preds = model.predict(sequences)  # Model predictions
    y_pred.extend(np.argmax(preds, axis=1))  # Get the predicted class (fight=1, non_fight=0)

# Compute metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.show()
