# **Real-Time American Sign Language (ASL) Detection System**

*🌟 Empowering Accessibility Through Technology: A Real-Time ASL Detection System 🌟*


In the world of accessibility, technology has the power to change lives—and this project does just that. This journey has been transformative, blending machine learning, deep learning, and computer vision into something meaningful: a Real-Time American Sign Language (ASL) Detection System. This system, developed with OpenCV and CNN-based deep learning, bridges communication gaps and showcases how technology can assist the deaf and hard-of-hearing community.


# Table of Contents-

  1.Project Overview

  2.Uses and Significance

  3.How the Project Works

  4.Installation

  5.Directory Structure

  6.Data Collection
  
  7.Model Training

  8.Real-Time Prediction

  9.Future Plans

  10.Acknowledgments
  

# Project Overview-

This project was designed to identify ASL hand signs from A-Z in real time, enabling seamless communication through a live video feed. Leveraging TensorFlow and OpenCV, this model captures subtle movements of hands essential to ASL, a complete language with its own grammar independent of spoken English.


# Key Features-

Real-Time ASL Detection: Detects and classifies hand signs in real-time.
Efficient Model Training: A CNN optimized for hand gesture classification.
Accessible Codebase: Structured and easy-to-follow notebooks.


# Uses and Significance-

In a world where communication is often taken for granted, this system promotes inclusivity, enabling people who are deaf or hard of hearing to communicate more effectively. By automating ASL recognition, this project has the potential to empower users, create accessible interactions, and raise awareness about ASL’s complexity as a linguistic system.


# How the Project Works-

This project is broken down into three main components:

**Data Collection**: Custom datasets of ASL signs were built from scratch. The data collection notebook enables users to capture hand gestures for each letter in varied lighting and angles to ensure robust model training.

**Model Training**: The Convolutional Neural Network (CNN) uses TensorFlow to recognize patterns and classify hand gestures. This model has been carefully optimized with hyperparameter tuning, EarlyStopping, and ReduceLROnPlateau for achieving over 80% accuracy.

**Real-Time Prediction**: By combining OpenCV with the trained model, real-time predictions are made directly on the video feed. Each frame is processed and interpreted, displaying the predicted ASL character on-screen.


# Installation-

**Prerequisites-**

Ensure you have Python 3.7+ installed on your machine. Use the following command to install required libraries:

pip install opencv-python tensorflow numpy matplotlib



**Required Libraries-**

*OpenCV (cv2)*: For capturing video feeds and real-time image processing.

*TensorFlow & Keras*: For CNN model building and deployment.

*NumPy*: For efficient data handling.

*Matplotlib*: For visualizing model training results.


# Directory Structure-

Ensure your project directory is organized as follows:


├── data

│   └── images                  ( Captured images, organized by hand sign (A-Z) )

├── datacollection.ipynb       ( Notebook for collecting hand sign data )

├── CNNmodeltraining.ipynb     ( Notebook for training the CNN model )

├── prediction.ipynb           ( Notebook for real-time prediction )

└── README.md


# Data Collection-

**To start building the dataset:**

Open datacollection.ipynb: This notebook uses the webcam to capture images for each hand gesture.

Label Data: Assign the correct label to each gesture, saved in data folder.

Diversity: Capture gestures under various lighting conditions to improve model generalizability.


# Model Training-

**To train the ASL detection model:**

Open CNNmodeltraining.ipynb.

Data Loading: The notebook loads and preprocesses images, organizing them into training and validation sets.

Train the CNN: The model learns to classify gestures based on the input images.

**Model Architecture-**

The CNN consists of several convolutional layers for feature extraction, followed by fully connected layers for classification. Hyperparameters have been fine-tuned to achieve efficient model performance without overfitting.


# Real-Time Prediction-

**To run the model in real-time:**

Open prediction.ipynb.

Model Loading: This notebook loads the trained model for real-time prediction.

Video Feed Prediction: OpenCV processes each frame, using the CNN to classify hand gestures. Predictions are displayed directly on the video feed.


# Future Plans-

**This project is the foundation for a broader vision:**

*Expanding Dataset*: Future iterations will include more complex ASL signs, words, and phrases.

*Mobile and Embedded Deployment*: Plans to optimize the model for mobile devices, making it more accessible to users in real-world settings.


# Acknowledgments-

This project was inspired by the profound impact that accessible technology can have on people's lives. Celebrating International Week of the Deaf highlighted the potential of this work in bridging communication gaps and fostering inclusivity.
