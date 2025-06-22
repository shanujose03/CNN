# Face Recognition

This is a face recognition system using deep learning techniques with OpenCV, Keras, and Streamlit. The system captures face images, trains a custom CNN model, and recognizes faces in real-time through a web interface which ican be used for biometric authentication.

This system provides the following functionalities:

- Capture face images through webcam using OpenCV.
- Train a CNN model with Keras on the captured face data.
- Recognize faces in real-time using a Streamlit web application.
- Display the name of the recognized person.

This project uses OpenCV's Haar Cascade Classifier for face detection.
The classifier file used: haarcascade_frontalface_default.xml

File Descriptions:

 **face.py**: Uses OpenCV to capture images from the webcam and save detected face regions.
**model.py**: Defines and trains a CNN model on the captured images; saves the trained model.
**app.py**: Provides a web interface with Streamlit for real-time face recognition.
**face_model.keras**: The trained CNN model file.
**requirements.txt**: List of required Python packages.



