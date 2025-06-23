# Emotion-tracking-
Emotions on face detected using Convolutional Neural Network model.
This project implements a facial emotion recognition system using a deep learning model based on MobileNetV2. The system is trained to classify facial expressions into multiple emotion categories and supports both offline image analysis and real-time emotion detection using a webcam. Additionally, Grad-CAM visualizations are generated to interpret model predictions.

Features
Trains a convolutional neural network (CNN) using transfer learning from MobileNetV2

Classifies facial expressions into seven emotions: disgust, fear, happy, neutral, sad, surprise, and unhappy

Augments image data and uses validation split for improved training performance

Saves prediction logs and visual distribution of predictions

Uses Grad-CAM to visualize which parts of the image influenced the model's decisions

Supports real-time facial emotion detection using webcam input and OpenCV

Requirements
Python 3.x

TensorFlow 2.x

OpenCV

NumPy

Matplotlib

Pandas

To install the required packages:

nginx
Copy
Edit
pip install tensorflow opencv-python numpy matplotlib pandas
Dataset Structure
The dataset should be organized into subdirectories, each named after an emotion label. Each subdirectory must contain face images corresponding to that emotion.

Copy
Edit
project_directory/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
├── surprise/
├── unhappy/
Each folder can contain .jpg, .jpeg, or .png images.

How It Works
The script loads and augments the dataset using ImageDataGenerator.

A CNN model is built using MobileNetV2 as the base model. The base model’s weights are frozen.

A new classification head is added with fully connected layers and a softmax output.

The model is trained and evaluated. Predictions are logged into a CSV file.

Grad-CAM visualizations are generated for one image from each emotion class.

The trained model is saved to disk.

A real-time detection script loads the model, detects faces from webcam input, and displays predicted emotions live.

Output
emotion_mobilenet_model.h5: Trained Keras model file

mobilenet_emotion_logs.csv: Prediction log with timestamps, true labels, and confidence scores

mobilenet_emotion_distribution.png: Bar chart showing distribution of predicted emotions

Real-Time Detection
After training the model, the script can activate a webcam to detect faces and classify emotions in real-time. The detected face is processed and passed through the model to predict the emotion. The result is displayed along with a bounding box and confidence score.

To exit the webcam feed, press the q key.

Notes
Ensure your webcam is connected and accessible.

You can run the real-time detection code separately after training using the saved .h5 model.

Grad-CAM is applied to only the first image in each class to avoid unnecessary computation.

The model is portable and can be used on any compatible device with TensorFlow installed.
