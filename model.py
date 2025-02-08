import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras import layers, models
#from keras.utils import to_categorical, Sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.np_utils import to_categorical


# Data Loader Class (Handles multiple `.npy` files per video)
class SequenceDataset(Sequence):
    def __init__(self, sequences_dir, batch_size=32, sequence_length=10, target_size=(299, 299), num_classes=2):
        self.sequences_dir = sequences_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.num_classes = num_classes
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        """Loads sequences from nested directories, ensuring multiple `.npy` files in each video folder are read."""
        for class_name in ['fight', 'non_fight']:
            class_path = os.path.join(self.sequences_dir, class_name)
            label = 1 if class_name == "fight" else 0

            if os.path.exists(class_path):
                for video_folder in os.listdir(class_path):
                    video_folder_path = os.path.join(class_path, video_folder)

                    if os.path.isdir(video_folder_path):
                        for sequence_file in sorted(os.listdir(video_folder_path)):
                            sequence_path = os.path.join(video_folder_path, sequence_file)
                            if sequence_path.endswith('.npy'):
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
            sequence = np.array([cv2.resize(frame, self.target_size) for frame in sequence])
            sequences.append(sequence)

        sequences = np.array(sequences)  # Shape: [batch_size, sequence_length, height, width, channels]
        batch_labels = np.array(batch_labels)
        return sequences, to_categorical(batch_labels, num_classes=self.num_classes)

#  Build Model (Xception + Bi-LSTM + Attention)
def build_model(sequence_length, input_shape=(299, 299, 3), num_classes=2):
    # CNN Feature Extractor (Xception)
    cnn_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    cnn_base.trainable = False

    cnn_output = layers.GlobalAveragePooling2D()(cnn_base.output)
    cnn_model = models.Model(inputs=cnn_base.input, outputs=cnn_output)

    # Frame Sequence Input
    input_seq = layers.Input(shape=(sequence_length, *input_shape))
    cnn_features = layers.TimeDistributed(cnn_model)(input_seq)

    # Bi-LSTM Layer
    lstm_out = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(cnn_features)

    # Attention Mechanism
    attention = layers.Attention()([lstm_out, lstm_out])
    attention_pooled = layers.GlobalAveragePooling1D()(attention)

    # Fully Connected Layer
    output = layers.Dense(num_classes, activation='softmax')(attention_pooled)

    # Define Model
    model = models.Model(inputs=input_seq, outputs=output)
    return model

#  Initialize Model
sequence_length = 10
input_shape = (299, 299, 3)
model = build_model(sequence_length, input_shape=input_shape)

#  Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Load Datasets
train_dataset = SequenceDataset("D:\\project_folder\\final_dataset\train", batch_size=32)
val_dataset =   SequenceDataset("D:\\project_folder\\final_dataset\\validate", batch_size=32)
test_dataset =  SequenceDataset("D:\\project_folder\\final_dataset\\test", batch_size=32)


#  Train Model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

#  Save Model
model.save('fight_detection_xception_bilstm_attention.h5')

#  Evaluate Model
y_true = []
y_pred = []
for sequences, labels in test_dataset:
    preds = model.predict(sequences)
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

#  Performance Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n🔍 **Model Performance Metrics**")
print(f" Accuracy: {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1 Score: {f1:.4f}")

#  Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fight', 'Fight'], yticklabels=['Non-Fight', 'Fight'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

#  Training History Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()
