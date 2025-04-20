import os
import numpy as np
import tensorflow as tf
import cv2
from keras import layers, models, preprocessing, utils
from keras.src.utils import to_categorical


# Define the CNN + Bi-LSTM Model
class CNN_BiLSTM_Model(tf.keras.Model):
    def __init__(self, sequence_length, num_classes=2):
        super(CNN_BiLSTM_Model, self).__init__()

        # CNN feature extractor using pre-trained ResNet50
        self.cnn = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        self.cnn.trainable = False  # Freeze the layers of the CNN

        # Global Average Pooling to flatten the output of CNN
        self.global_pool = layers.GlobalAveragePooling2D()

        # BiLSTM layer
        self.lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=False))

        # Fully connected layer for final classification
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Extract features using CNN
        batch_size, sequence_length, height, width, channels = inputs.shape  # Get the static shape

        cnn_out = []
        for i in range(sequence_length):
            frame = inputs[:, i, :, :, :]  # Extract a frame from the sequence
            cnn_feature = self.cnn(frame)  # Get CNN features
            pooled_feature = self.global_pool(cnn_feature)  # Apply global average pooling
            cnn_out.append(pooled_feature)

        cnn_out = tf.stack(cnn_out, axis=1)  # Stack CNN features to form sequence

        # Pass the CNN features through BiLSTM
        lstm_out = self.lstm(cnn_out)  # Process sequence with LSTM

        # Classify the output using the fully connected layer
        output = self.fc(lstm_out)

        return output



# Dataset loading and preprocessing
class SequenceDataset(tf.keras.utils.Sequence):
    def __init__(self, sequences_dir, batch_size=32, sequence_length=10, target_size=(299, 299)):
        self.sequences_dir = sequences_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for class_name in os.listdir(self.sequences_dir):  # 'fight', 'non_fight'
            class_path = os.path.join(self.sequences_dir, class_name)
            label = 1 if class_name == "fight" else 0

            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                for sequence_file in os.listdir(video_path):
                    sequence_path = os.path.join(video_path, sequence_file)
                    self.data.append(sequence_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        sequences = []
        for seq_path in batch_data:
            sequence = np.load(seq_path)  # Load sequence (shape: [sequence_length, height, width, channels])
            sequence = np.array([cv2.resize(frame, self.target_size) for frame in sequence])  # Resize frames
            sequences.append(sequence)

        sequences = np.array(sequences)  # Shape: [batch_size, sequence_length, height, width, channels]
        batch_labels = np.array(batch_labels)
        return sequences, to_categorical(batch_labels, num_classes=2)


# Initialize model, loss function, and optimizer
sequence_length = 10
model = CNN_BiLSTM_Model(sequence_length=sequence_length)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create DataLoaders (train, validation)
train_dataset = SequenceDataset("D:\\project_folder\\final_dataset\\train", batch_size=32, sequence_length=sequence_length)
val_dataset = SequenceDataset("D:\\project_folder\\final_dataset\\validate", batch_size=32, sequence_length=sequence_length)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=1)

# Save the trained model
model.save('cnn_bilstm_fight_detection_model.h5')