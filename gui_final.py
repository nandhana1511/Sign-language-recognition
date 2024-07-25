import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import json
import yt_dlp

# Load the trained model for image recognition
image_model = tf.keras.models.load_model('C:\\Users\\DELL\\Desktop\\Sign Language Recognition\\sl_image_model.h5')

# Create a dictionary to map the model's output to the corresponding letters
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def classify_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = image_model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        
        result = classify_image(file_path)
        result_label.config(text=f'Recognized Sign: {result}')

# Load the trained model for video recognition
def download_video_from_youtube(youtube_url, output_path='downloaded_video.mp4'):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,
        'noplaylist': True,
        'ffmpeg_location': 'C:\\Users\\DELL\\Downloads\\ffmpeg-master-latest-win64-gpl\\ffmpeg-master-latest-win64-gpl\\bin',
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path

def predict_sign_language_from_video(video_path, model_path='sign_language_model.h5', label_mapping_path='sign_language_data/label_mapping.json'):
    model = tf.keras.models.load_model(model_path)
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) // 2
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame_number % frame_rate == 0:
            frame = cv2.resize(frame, (64, 64)) / 255.0
            frames.append(frame)

    cap.release()
    frames = np.array(frames)

    predictions = model.predict(frames)
    predicted_labels = np.argmax(predictions, axis=1)

    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    majority_label = unique_labels[np.argmax(counts)]

    return reverse_label_mapping[majority_label]

def on_predict():
    url = url_entry.get()
    if not url:
        messagebox.showwarning("Input Error", "Please enter a YouTube URL.")
        return
    
    try:
        video_path = download_video_from_youtube(url)
        gloss = predict_sign_language_from_video(video_path)
        result_label.config(text=f'Predicted Gloss: {gloss}')
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
root = tk.Tk()
root.title('Sign Language Recognition')

# Create and place the widgets for image recognition
upload_button = tk.Button(root, text='Upload Image', command=upload_image)
upload_button.pack(pady=5)

image_label = tk.Label(root)
image_label.pack(pady=5)

result_label = tk.Label(root, text='Recognized Sign: ')
result_label.pack(pady=5)

# Create and place the widgets for video recognition
tk.Label(root, text="Enter YouTube URL:").pack(pady=5)
url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=5)

predict_button = tk.Button(root, text="Predict", command=on_predict)
predict_button.pack(pady=20)

# Run the GUI event loop
root.mainloop()
