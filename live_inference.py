import os
import sys

print("[INFO] Initializing Live Inference Engine...", flush=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

def start_production_monitor():
    model_path = Path("mobilenet_finetuned_mask_model.keras")
    
    if not model_path.exists():
        print(f"[ERROR] Model artifact not found at: {model_path.resolve()}.")
        return
    
    print(f"[INFO] Loading fine-tuned model from: {model_path.resolve()}", flush=True)
    model = tf.keras.models.load_model(model_path)
    img_size = model.input_shape[1:3]  # (224, 224)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print(f"[ERROR] Failed to load Haar Cascade classifier.")
        return

    print("[INFO] Accessing webcam...", flush=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Unable to access webcam.")
        return

    print("[INFO] Live monitor active. Press 'q' inside the video window to exit.", flush=True)
    
    class_labels = ["With Mask", "Without Mask"]
    
    # =========================================================================
    # FIX FOR FLIPPED LABELS: 
    # If it says "With Mask" when you have no mask, change this to True.
    # If it says "Without Mask" when you are wearing a mask, change this to False.
    # =========================================================================
    INVERT_LABELS = True  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
                
            face_resized = cv2.resize(face_roi, img_size)
            face_array = tf.keras.utils.img_to_array(face_resized)
            face_array = np.expand_dims(face_array, axis=0)
            face_processed = preprocess_input(face_array)
            
            predictions = model.predict(face_processed, verbose=0)[0]
            pred_idx = np.argmax(predictions)
            confidence = predictions[pred_idx] * 100
            
            # Apply label inversion flag if classes are flipped
            if INVERT_LABELS:
                pred_idx = 1 - pred_idx
                
            label = class_labels[pred_idx]
            
            # Strict Color Logic: Without Mask = Red, With Mask = Green
            if "without" in label.lower():
                box_color = (0, 0, 255)   # Red
            else:
                box_color = (0, 255, 0)   # Green
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            
            display_text = f"{label}: {confidence:.1f}%"
            text_y = y - 10 if y - 10 > 20 else y + h + 25
            
            cv2.putText(
                frame, 
                display_text, 
                (x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                box_color, 
                2, 
                cv2.LINE_AA
            )

        cv2.imshow("Fine-Tuned MobileNetV2 - Live Mask Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam session closed.")

if __name__ == "__main__":
    start_production_monitor()