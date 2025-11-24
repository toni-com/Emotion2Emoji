# Emotion2Emoji: Real-Time Facial Expression Recognition

**Emotion2Emoji** is a Computer Vision application that detects human faces in images and classifies their emotion into a corresponding Emoji.

Built with **PyTorch**, it features a custom-trained ResNet50 model, a robust inference pipeline capable of handling real-world noise, and a clean visualization engine.

## Results

The model was trained on the FER-2013 dataset and generalizes well to unseen data.

| Happy (Keanu Reeves) | Angry (Sheldon Cooper) |
| :---: | :---: |
| *Accurate detection of a classic smile.* | *Detects intense gaze of anger.* |
| ![keanu](https://github.com/user-attachments/assets/55550abd-029a-4ec3-8730-e24a63587dcb) | ![sheldon](https://github.com/user-attachments/assets/224d2823-8470-4153-97ea-fd0a1af58096) |

| Multi-Face Detection (Linkin Park) | Console Inference |
| :---: | :---: |
| *Capable of batch processing multiple faces in a single frame.* | *Simple, scriptable command-line interface for batch jobs.* |
| ![linkin](https://github.com/user-attachments/assets/38f61164-3711-4d67-8940-f06e57136cc8)| <img width="1062" height="425" alt="ConsoleNeutral" src="https://github.com/user-attachments/assets/c49e1de8-1122-4f56-8822-638ca5222d3d" />|

## Key Features

* **Deep Learning Backbone:** Powered by a customized **ResNet50** architecture. The final classification head was modified to output 7 distinct emotion classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) instead of ImageNet's 1000.
* **Robust Preprocessing Pipeline:** Implements a ETL pipeline using `torchvision.transforms` to handle grayscale conversion, channel duplication (for ResNet compatibility), and normalization.
* **Smart Inference Engine:**
    * **Primary Detector:** Uses OpenCV's Haar Cascades or YuNet for fast face detection.
* **Training Dynamics:** Features a learning rate scheduler (`ReduceLROnPlateau`) and checkpointing system to capture the best model weights during training, preventing overfitting.

## Technology Stack

* **PyTorch & Torchvision:** For model architecture, dataset management, and training loops.
* **OpenCV (cv2):** For image I/O, face detection, and real-time drawing.

