import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
dataset_path = "images/train/Shanu"
class_name = os.path.basename(dataset_path)  # Extract 'Shanu'
model = load_model("face_model.keras")
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
st.title("Face Recognition")
frame_window = st.image([])
camera = cv2.VideoCapture(0)
run = st.button("Start Camera")

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera not found.")
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(100,100))
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        face_img = frame[y:y+h, x:x+w]
        face_img=cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (128, 128))    
        face_img = img_to_array(face_img) / 255.0 
        face_img = np.expand_dims(face_img, axis=0)
        prediction = model.predict(face_img)[0][0]
        print(f"Prediction Score: {prediction}")
        threshold=0.5
        if prediction < threshold:
            label = f"Hello {class_name}"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()


 

