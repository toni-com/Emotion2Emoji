# Emotion2Emoji: Real-Time Facial Expression Recognition

**Emotion2Emoji** is a Computer Vision application that detects human faces in images and classifies their emotion into a corresponding Emoji.

Built with **PyTorch**, it features a custom-trained ResNet50 model, a robust inference pipeline capable of handling real-world noise, and a clean visualization engine.

## Results

The model was trained on the FER-2013 dataset and generalizes well to unseen data.

| Happy (Keanu Reeves) | Angry (Sheldon Cooper) |
| :---: | :---: |
| *Accurate detection of a classic smile.* | *Detects intense gaze of anger.* |
| ![keanuhappy](https://github.com/user-attachments/assets/86174876-7525-4d72-9919-bfdf6060edf7) | ![ResultAngry](https://github.com/user-attachments/assets/5218d7b8-44f0-4249-9f2d-2533fa27b4a4) |

| Multi-Face Detection (Linkin Park) | Console Inference |
| :---: | :---: |
| *Capable of batch processing multiple faces in a single frame.* | *Simple, scriptable command-line interface for batch jobs.* |
| ![linkingneutral](https://github.com/user-attachments/assets/52576579-3cf9-4833-9a1e-81dbbeeb4e36)| <img width="1062" height="425" alt="ConsoleNeutral" src="https://github.com/user-attachments/assets/c49e1de8-1122-4f56-8822-638ca5222d3d" />|

## Key Features

* **Deep Learning Backbone:** Powered by a customized **ResNet50** architecture. The final classification head was surgically modified to output 7 distinct emotion classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) instead of ImageNet's 1000.
* **Robust Preprocessing Pipeline:** Implements a professional ETL pipeline using `torchvision.transforms` to handle grayscale conversion, channel duplication (for ResNet compatibility), and normalization.
* **Smart Inference Engine:**
    * **Primary Detector:** Uses OpenCV's Haar Cascades or YuNet for fast face detection.
    * **Fallback Logic:** Includes a heuristic fallback system that allows the model to process "difficult" faces (like Anime/Cartoons) even when standard face detectors fail.
* **Training Dynamics:** Features a learning rate scheduler (`ReduceLROnPlateau`) and checkpointing system to capture the best model weights during training, preventing overfitting.

## Technology Stack

* **PyTorch & Torchvision:** For model architecture, dataset management, and training loops.
* **OpenCV (cv2):** For image I/O, face detection, and real-time drawing.

