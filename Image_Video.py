import cv2
import os
import torch
import gradio as gr
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from collections import Counter
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN

# Load model and processor
processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")

# Load face detection model
face_detector = MTCNN(keep_all=True, device='cpu')

def calculate_confidence(logits):
    probs = F.softmax(logits, dim=-1)
    confidence = torch.max(probs).item() * 100
    return confidence

def get_manipulation_nature(image, prediction):
    if prediction == 0:  # If it's classified as real
        return "No manipulation detected"

    # Convert PIL Image to numpy array
    img_np = np.array(image)
    
    # Detect faces
    boxes, _ = face_detector.detect(img_np)
    
    if boxes is None:
        return "No face detected for analysis"

    manipulations = []

    for box in boxes:
        x, y, w, h = [int(b) for b in box]
        face = img_np[y:y+h, x:x+w]

        # Check for unusual color distributions (potential sign of GAN artifacts)
        hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
        if np.std(hsv[:,:,1]) > 50:  # High saturation variance
            manipulations.append("Unusual color patterns")

        # Check for blurriness (potential sign of image tampering)
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 100:
            manipulations.append("Blurry regions")

        # Check for inconsistent noise patterns
        noise = cv2.fastNlMeansDenoisingColored(face, None, 10, 10, 7, 21)
        diff = cv2.absdiff(face, noise)
        if np.mean(diff) > 10:
            manipulations.append("Inconsistent noise patterns")

    if not manipulations:
        manipulations.append("Subtle manipulations (details unclear)")

    return f"Potential manipulations detected: {', '.join(set(manipulations))}"

def process_image(image):
    image_pil = Image.open(image).convert("RGB")
    inputs = processor(images=image_pil, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    confidence = calculate_confidence(outputs.logits)
    
    manipulation_nature = get_manipulation_nature(image_pil, prediction)
    
    return ("fake" if prediction == 1 else "real", image, confidence, manipulation_nature)

def process_video(video_path, progress=gr.Progress()):
    cap = cv2.VideoCapture(video_path.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(fps // 4))
    predictions = []
    confidences = []
    manipulation_natures = []
    frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress(0, desc="Processing video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            confidence = calculate_confidence(outputs.logits)
            
            manipulation_nature = get_manipulation_nature(image, prediction)
            
            predictions.append(prediction)
            confidences.append(confidence)
            manipulation_natures.append(manipulation_nature)
            progress(frame_count / total_frames, desc="Processing video")

        frame_count += 1

    cap.release()

    mode_prediction = Counter(predictions).most_common(1)[0][0]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0
    most_common_manipulation = Counter(manipulation_natures).most_common(1)[0][0]
    
    return ("fake" if mode_prediction == 1 else "real", video_path, mean_confidence, most_common_manipulation)

# ... [rest of the code remains the same] ...

if __name__ == "__main__":
    demo.launch(share=True)