import argparse

import torch
import cv2
from PIL import Image
from torchvision import transforms

from src.model import FaceModel


def main(path: str, version: str = "data/models/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceModel(pretrained=False)
    model.load_state_dict(torch.load(version, map_location=device))
    model.to(device)
    model.eval()

    #
    # image loading
    #
    original_image = cv2.imread(path)
    if original_image is None:
        print("Error: Image not found or unable to read.")
    else:
        print("Image loaded successfully!")

    #
    # face detection
    #
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_coords = face_classifier.detectMultiScale(  # <- returns coordinates
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(face_coords) == 0:
        print("\n No face detected.")
        print("\n Switch to Fallback: Using the entire image as the face.")

        h, w = original_image.shape[:2]
        face_coords = [[0, 0, w, h]]
    else:
        print(f"Found {len(face_coords)} faces!")

    #
    # transform just like train data
    #
    test_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    emoji_map = {0: "😠", 1: "🤢", 2: "😨", 3: "😊", 4: "😐", 5: "😢", 6: "😲"}

    for x, y, w, h in face_coords:
        face_roi = original_image[y : y + h, x : x + w]

        # Preprocess for Model
        # Convert numpy -> PIL
        pil_face = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        # add another dimension so that -> batch dimension [1, 3, 224, 224]
        face_tensor = test_transform(pil_face).unsqueeze(0).to(device)

        # actual Inference
        with torch.no_grad():
            output = model(face_tensor)
            pred_idx = torch.argmax(output).item()
            emoji = emoji_map[pred_idx]

        # D. Draw on Image (The Wow Factor)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            original_image,
            str(pred_idx),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )

        print(f"Face at ({x},{y}): {emoji}")

    cv2.imwrite("prediction_result.jpg", original_image)
    print("Saved result to prediction_result.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--model", default="data/models/best_model.pth")

    args = parser.parse_args()
    main(path=args.path, version=args.model)
