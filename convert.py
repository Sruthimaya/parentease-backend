"""
Baby Cry Detection + ESP32 Audio Recording + AI Prediction (Optimized)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import socket
import time
import librosa
import sys
import tensorflow as tf
import keras
from keras import layers as keras_layers

# Provide LocallyConnected2D symbol for DeepFace / legacy models
try:
    LocallyConnected2D = keras_layers.LocallyConnected2D
except AttributeError:
    class LocallyConnected2D(keras_layers.Layer):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._conv = keras_layers.Conv2D(
                filters=kwargs.get("filters", 1),
                kernel_size=kwargs.get("kernel_size", (1, 1)),
                strides=kwargs.get("strides", (1, 1)),
                padding="valid",
                activation=kwargs.get("activation", None),
                use_bias=kwargs.get("use_bias", True),
            )

        def call(self, inputs):
            return self._conv(inputs)

import tensorflow.keras.layers as tf_keras_layers
setattr(tf_keras_layers, "LocallyConnected2D", LocallyConnected2D)
sys.modules["tensorflow.keras.layers"] = tf_keras_layers

from deepface import DeepFace
import firebase_admin
from firebase_admin import credentials, db

# -------------------------------
# ESP32 SETTINGS
# -------------------------------

ESP32_IP = "192.168.215.38"
PORT = 5000

AUDIO_FILE = "received_audio.wav"
SAMPLE_RATE = 16000

# -------------------------------
# FIREBASE
# -------------------------------

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://infantcare-de72f-default-rtdb.firebaseio.com/'
    })

# -------------------------------
# LOAD AI MODEL
# -------------------------------

print("Loading audio AI model...")
model = keras.models.load_model("resnet_audio_model.h5", compile=False)
print("Model loaded")
classes = [
    'belly pain',
    'burping',
    'cold_hot',
    'discomfort',
    'tired'
]

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------

def extract_feature(file):

    audio, sr = librosa.load(file, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel)

    if mel_db.shape[1] < 128:
        pad = 128 - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode='constant')

    mel_db = mel_db[:, :128]

    mel_db = mel_db[..., np.newaxis]
    mel_db = np.repeat(mel_db, 3, axis=-1)
    mel_db = mel_db[np.newaxis,...]

    return mel_db

# -------------------------------
# AUDIO PREDICTION
# -------------------------------

def predict_audio(file):

    print("\nRunning AI prediction...")

    feature = extract_feature(file)

    prediction = model.predict(feature, verbose=0)

    index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    predicted_class = classes[index]

    print("Prediction:", predicted_class)
    print("Confidence:", round(confidence,2), "%")

    data = {
        "prediction": predicted_class,
        "confidence": float(round(confidence,2)),
        "timestamp": int(time.time()),
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    ref = db.reference("baby_monitor")
    ref.push(data)

    print("Result sent to Firebase")

# -------------------------------
# ESP32 COMMUNICATION
# -------------------------------

# ------------------------------- 
# ESP32 COMMUNICATION (Fixed)
# -------------------------------

def send_cry_command():

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(15)  # increased timeout

    try:
        print("Connecting ESP32...")
        client.connect((ESP32_IP, PORT))

        # Send command
        client.sendall(b"CRY_DETECTED\n")

        # --- READ READY SIGNAL AS TEXT ---
        ready_bytes = b""
        while b"\n" not in ready_bytes:
            chunk = client.recv(1)
            if not chunk:
                raise Exception("No READY signal from ESP32")
            ready_bytes += chunk

        ready = ready_bytes.decode("utf-8").strip()
        if ready != "READY":
            print("ESP32 did not respond with READY")
            return

        print("ESP32 READY → Receiving audio")

        # --- RECEIVE AUDIO AS BINARY ---
        total_bytes = 0
        with open(AUDIO_FILE, "wb") as f:
            while True:
                try:
                    data = client.recv(4096)
                    if not data:
                        break
                    f.write(data)
                    total_bytes += len(data)
                    print("Receiving:", total_bytes, "bytes", end="\r")
                except socket.timeout:
                    print("\nSocket timeout reached")
                    break

        print("\nAudio received")

        if os.path.exists(AUDIO_FILE) and os.path.getsize(AUDIO_FILE) > 100:
            predict_audio(AUDIO_FILE)

    except Exception as e:
        print("ESP32 connection error:", e)

    finally:
        client.close()
# -------------------------------
# FACE DETECTION
# -------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------
# WEBCAM
# -------------------------------

cap = cv2.VideoCapture(0)

last_trigger = 0
trigger_delay = 15

frame_count = 0
skip_frames = 5

print("\nSystem Started")
print("Press Q to quit\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    frame_count += 1

    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]

        try:

            if frame_count % skip_frames != 0:
                continue

            result = DeepFace.analyze(
                face,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            emotions = result["emotion"]

            dominant = max(emotions, key=emotions.get)

            crying_score = (
                emotions.get("sad",0) +
                emotions.get("angry",0) +
                emotions.get("fear",0)
            )

            crying_score = crying_score / 100

            is_crying = crying_score > 0.5

            color = (0,0,255) if is_crying else (0,255,0)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            text = f"{dominant} {crying_score*100:.1f}%"

            cv2.putText(
                frame,
                text,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            if is_crying:

                current_time = time.time()

                if current_time - last_trigger > trigger_delay:

                    print("\nCRY DETECTED → Triggering ESP32")

                    send_cry_command()

                    last_trigger = current_time

        except Exception as e:
            print("Emotion error:", e)

    cv2.imshow("Baby Cry Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()