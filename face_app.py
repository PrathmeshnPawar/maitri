import cv2
from deepface import DeepFace

# Read an image from file
img = cv2.imread("test_image_2.jpg")
try:
    result = DeepFace.analyze(img, actions=['emotion'])
    print("DeepFace analysis successful.")
    print(result)
except Exception as e:
    print(f"DeepFace failed: {e}")