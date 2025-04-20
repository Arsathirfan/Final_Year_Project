import tkinter as tk
from tkinter import ttk
import pygame
import time
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.lite as tflite
import pyaudio
import threading
from PIL import Image, ImageTk
import notification
from tkinter import messagebox

violence_count = 0
phone_numbers = [
    # "9876543210",
]

root = tk.Tk()
root.title("Surveillance System")
root.geometry("700x550")
root.configure(bg="#2c3e50")

pygame.mixer.init()

alarm_playing = False
alarm_start_time = 0

video_model = tf.keras.models.load_model("modelnew.h5")

# Load the audio fight detection model
interpreter = tflite.Interpreter(model_path="audio_model.tflite")
interpreter.allocate_tensors()

# Get model input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Audio parameters
SAMPLE_RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Expected audio input shape
input_shape = input_details[0]['shape']
EXPECTED_SAMPLES = input_shape[1]  # 44032 samples

# Load class labels for audio
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Define image size (should match what was used during training)
IMG_SIZE = 128

# Define labels for video model
VIDEO_CLASSES = ["No Fight", "Fight"]

# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Start microphone stream
pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                 input=True, frames_per_buffer=EXPECTED_SAMPLES)

# Initialize surveillance state
surveillance_running = False
video_label = ""
audio_label = ""
video_confidence = 0.0
lock = threading.Lock()


# Function to process audio and run inference
def predict_audio_fight():
    global audio_label
    while surveillance_running:
        audio_data = stream.read(EXPECTED_SAMPLES, exception_on_overflow=False)
        audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Ensure correct size (zero-pad or trim)
        if len(audio_data) < EXPECTED_SAMPLES:
            audio_data = np.pad(audio_data, (0, EXPECTED_SAMPLES - len(audio_data)))
        else:
            audio_data = audio_data[:EXPECTED_SAMPLES]

        # Reshape to match model input shape (1, 44032)
        audio_data = np.expand_dims(audio_data, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], audio_data.astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction
        predicted_label = labels[np.argmax(output_data)]
        confidence = np.max(output_data)

        print("Audio Prediction:", predicted_label)

        with lock:
            audio_label = predicted_label


# Function to process video and run inference
def predict_video_fight():
    global video_label, video_confidence
    while surveillance_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for video model
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb / 255.0
        frame_expanded = np.expand_dims(frame_normalized, axis=0)

        # Video model prediction
        video_prediction = video_model.predict(frame_expanded)[0][0]
        video_label_pred = VIDEO_CLASSES[1] if video_prediction > 0.5 else VIDEO_CLASSES[0]
        video_confidence = video_prediction if video_prediction > 0.5 else 1 - video_prediction

        #print("Video Prediction:", video_label_pred)

        with lock:
            video_label = video_label_pred


# Initialize the alarm functionality
def play_alarm():
    global alarm_playing, alarm_start_time
    try:
        pygame.mixer.music.load("alaram_sound.mp3")  # Load the alarm sound
        pygame.mixer.music.play(-1)  # Play the alarm in loop
        alarm_playing = True
        alarm_start_time = time.time()  # Mark the time when the alarm starts
        print("Alarm is playing!")
    except pygame.error as e:
        print(f"Error loading alarm sound: {e}")


def stop_alarm():
    global alarm_playing
    pygame.mixer.music.stop()
    alarm_playing = False


def update_video_feed():
    ret, frame = cap.read()
    if ret:
        # Convert the image to Tkinter format
        frame_resized = cv2.resize(frame, (640, 480))  # Ensure the frame is large enough
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the image in the Tkinter window
        video_label_widget.config(image=img_tk)
        video_label_widget.image = img_tk

    # Check for violence detection (audio and video)
    check_violence()

    if surveillance_running:
        video_label_widget.after(10, update_video_feed)  # Continue updating every 10ms


# Function to monitor violence and trigger alarm
last_violence_time = time.time()

# Global flag to track if a notification has been sent
notification_sent = False

def check_violence():
    global violence_count, last_violence_time, notification_sent  # Declare notification_sent as global

    check_camera_button.pack(pady=10)

    with lock:
        if video_label == "Fight" and audio_label == "Fight":
            print("⚠️ Real Violence Detected! ⚠️")
            violence_count += 1  # Increment the violence count
            last_violence_time = time.time()  # Update the last violence detection time
            notification_sent = False  # Reset the notification flag when a new detection occurs

        if violence_count >= 50 and not notification_sent:
            print("⚠️ Alarm Activated! ⚠️")
            # Send alert notifications to all phone numbers
            #notification.send_alert_notification(8778015076)
            # print(phone_numbers[0])
            time.sleep(2)
            if len(phone_numbers) > 0:
                notification.send_alert_notification(phone_numbers[0])
            else:
                print("No phone numbers available to send notifications.")

            if not alarm_playing:
                play_alarm()  # Trigger the alarm only after 50 detections
                violence_count = 0  # Reset violence count after alarm is triggered

            notification_sent = True  # Set the flag to true to prevent sending multiple notifications

        # Stop alarm after 10 seconds
        if alarm_playing and time.time() - alarm_start_time >= 10:
            stop_alarm()  # Stop the alarm after 10 seconds
            violence_count = 0  # Reset violence count after stopping the alarm

        # Reset the violence count if no violence is detected for 5 seconds
        if time.time() - last_violence_time > 5:
            if violence_count > 0:
                print("No violence detected for 5 seconds. Resetting violence count.")
                violence_count = 0  # Reset violence count after 5 seconds of no violence detected
                notification_sent = False  # Reset the notification flag as well


# Function to show video feed when "Check Camera" button is clicked
def show_video_feed():
    video_frame.pack(pady=20)


def hide_video_feed():
    video_frame.pack_forget()


def open_add_notifiers_window():
    # Create the popup window
    add_notifiers_window = tk.Toplevel(root)
    add_notifiers_window.title("Add Notifiers")
    add_notifiers_window.geometry("400x350")

    # Create a frame to contain the listbox
    listbox_frame = tk.Frame(add_notifiers_window)
    listbox_frame.pack(pady=10)

    # Create a listbox to show current phone numbers (with clear background for visibility)
    listbox = tk.Listbox(listbox_frame, height=10, width=40, bg="lightgray")
    listbox.pack()

    # Add current numbers to the listbox
    for number in phone_numbers:
        listbox.insert(tk.END, number)

    # Entry and Button for adding new numbers
    phone_number_label = tk.Label(add_notifiers_window, text="Enter Phone Number:")
    phone_number_label.pack(pady=10)

    phone_number_entry = tk.Entry(add_notifiers_window, width=30)
    phone_number_entry.pack(pady=5)

    def add_phone_number():
        new_number = phone_number_entry.get().strip()
        if new_number and new_number not in phone_numbers:
            phone_numbers.append(new_number)
            listbox.insert(tk.END, new_number)  # Update the listbox with the new number
            phone_number_entry.delete(0, tk.END)  # Clear the entry field
            tk.messagebox.showinfo("Success", "Phone Number Added Successfully!")  # Success message
        else:
            tk.messagebox.showwarning("Invalid Input", "Please enter a valid phone number or number already exists.")

    add_button = tk.Button(add_notifiers_window, text="Add", command=add_phone_number)
    add_button.pack(pady=10)

    # Function to remove selected phone number
    def remove_phone_number():
        selected_number_index = listbox.curselection()
        if selected_number_index:
            selected_number = listbox.get(selected_number_index)
            phone_numbers.remove(selected_number)
            listbox.delete(selected_number_index)
            tk.messagebox.showinfo("Removed", f"Phone number {selected_number} removed successfully!")
        else:
            tk.messagebox.showwarning("Selection Error", "Please select a phone number to remove.")

    # Create a context menu for right-click
    def show_context_menu(event):
        # Check if an item is selected
        selected_number_index = listbox.curselection()
        if selected_number_index:
            context_menu.post(event.x_root, event.y_root)  # Show the context menu at the mouse pointer

    # Create a context menu with a "Delete" option
    context_menu = tk.Menu(add_notifiers_window, tearoff=0)
    context_menu.add_command(label="Delete", command=remove_phone_number)

    # Bind right-click event to show context menu
    listbox.bind("<Button-3>", show_context_menu)

    # Button to close the window
    close_button = tk.Button(add_notifiers_window, text="Close", command=add_notifiers_window.destroy)
    close_button.pack(pady=10)


# Create the tkinter dialog box with enhanced UI
def create_gui():
    global start_button, stop_button, video_label_widget, check_camera_button, exit_button, video_frame

    # Frame to contain video feed
    video_frame = tk.Frame(root)
    video_label_widget = tk.Label(video_frame, bd=10, relief="solid", bg="#34495e")

    # Create Exit button inside the video frame (top-right corner)
    exit_button = tk.Button(video_frame, text="X", bg="red", fg="white", command=hide_video_feed)
    exit_button.place(x=600, y=10)  # Position it at the top-right corner of the video feed

    # Pack the video label widget inside the frame
    video_label_widget.pack(fill="both", expand=True)

    # Start and Stop buttons
    button_frame = tk.Frame(root, bg="#2c3e50")

    # Add a header label
    header_label = tk.Label(root, text="Surveillance System", font=("Helvetica", 20), bg="#2c3e50", fg="white")
    header_label.pack(pady=10)

    start_button = ttk.Button(button_frame, text="Start Surveillance", command=start_surveillance)
    start_button.pack(side=tk.LEFT, padx=20, pady=10)

    stop_button = ttk.Button(button_frame, text="Stop Surveillance", command=stop_surveillance, state=tk.DISABLED)
    stop_button.pack(side=tk.LEFT, padx=20, pady=10)

    button_frame.pack(pady=10)

    # Create the "Check Camera" button (visible only before surveillance starts)
    check_camera_button = tk.Button(root, text="Check Camera", bg="red", fg="white", command=show_video_feed)
    check_camera_button.pack(pady=10)

    # Disable "Check Camera" button once surveillance starts
    check_camera_button.config(state=tk.DISABLED)

    # Initialize the "Add Notifiers" button (only enabled before surveillance starts)
    add_notifiers_button = ttk.Button(root, text="Add Notifiers Number", command=open_add_notifiers_window)
    add_notifiers_button.pack(pady=10)

    # Set window background color
    root.configure(bg="#2c3e50")

    root.mainloop()

def start_surveillance():
    global surveillance_running
    surveillance_running = True
    video_thread = threading.Thread(target=predict_video_fight, daemon=True)
    audio_thread = threading.Thread(target=predict_audio_fight, daemon=True)
    video_thread.start()
    audio_thread.start()

    # Change Start button to Stop button
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

    check_camera_button.config(state=tk.NORMAL)  # Disable Check Camera button after starting surveillance
    update_video_feed()

# Stop surveillance process
def stop_surveillance():
    global surveillance_running
    surveillance_running = False
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    check_camera_button.config(state=tk.NORMAL)  # Enable Check Camera button when surveillance stops
    cap.release()
    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    pa.terminate()
    root.destroy()


# Start the GUI
create_gui()
