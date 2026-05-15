import cv2
import torch
import numpy as np
import argparse
import os
import pickle
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO  # Standard entry for YOLOv10/v11

# Import your custom model architecture
from models.base_block import TransformerClassifier
from config import argument_parser

# --- 1. Helper Functions of this inference ---

def load_uni_par_model(checkpoint_path, args):
    model = TransformerClassifier(args=args)
    if torch.cuda.is_available():
        model = model.cuda()
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Match the state dict keys based on your training log
    for key in ['model_state_dict', 'state_dicts', 'model']:
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
            break
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def preprocess_crop(cv2_image, height=256, width=128):
    """Prepares a single detection crop for the UniPAR model."""
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # Convert OpenCV BGR crop to PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    return transform(pil_img).unsqueeze(0)

def get_summary_text(predictions, attributes):
    """Replicates your logic for Gender and Age extraction."""
    gender_attr = 'female'
    age_attrs = ['age over 60', 'age 18 to 60', 'age less 18']
    
    gender = ""
    age_scores = []
    
    for attr, pred in zip(attributes, predictions):
        #print(f"\n\nDebugg: Predictions Before Deciding Gender: {attributes[0]}:{predictions[0]}")
        if attr.lower() == gender_attr.lower():
            gender = 'Female' if pred > 0.85 else 'Male'
            print(f"{gender}:{pred} \n\n")
        elif attr in age_attrs:
            age_scores.append(pred)
            
    best_age = age_attrs[np.argmax(age_scores)] if age_scores else "Unknown"
    return f"{gender}, {best_age}"

# --- 2. Main Video Inference Logic ---

def main():
    parser = argparse.ArgumentParser(description="UniPAR + YOLOv10 Video Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to UniPAR checkpoint')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--dataset', type=str, default='PA100k', help='Dataset name')
    parser.add_argument('--output', type=str, default='video_output/video_01.mp4')
    args = parser.parse_args()

    # Setup UniPAR Config
    config_args = argument_parser().parse_args([]) 
    config_args.dataset = [args.dataset]

    # Load Models
    print("Initializing models...")
    uni_par = load_uni_par_model(args.checkpoint, config_args)
    detector = YOLO("./weights/yolov10n.pt") # Downloads automatically if not found
    
    # Load Dataset info for attributes
    with open(f'./data/{args.dataset}/dataset.pkl', 'rb') as f:
        dataset_info = pickle.load(f)
    attributes = dataset_info.attr_words
    word_vec = torch.tensor(dataset_info.attr_vectors, dtype=torch.float32)
    if torch.cuda.is_available():
        word_vec = word_vec.cuda()

    # Video Setup
    cap = cv2.VideoCapture(args.video)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"Processing video: {args.video}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Step A: Detect Persons (Class 0 in COCO is Person)
        results = detector.predict(frame, classes=[0], verbose=False)[0]
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Step B: Crop Person
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0: continue

            # Step C: UniPAR Inference
            input_tensor = preprocess_crop(crop)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            with torch.no_grad():
                logits = uni_par(input_tensor, None, word_vec)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Step D: Parsing & Drawing
            label = get_summary_text(probs, attributes)
            
            # Draw Bounding Box and Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add a background for text for better readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        writer.write(frame)
        # cv2.imshow('UniPAR Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Finished! Video saved to {args.output}")

if __name__ == '__main__':
    main()