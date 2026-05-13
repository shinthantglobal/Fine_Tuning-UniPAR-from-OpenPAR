import argparse
import os
import pickle
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from models.base_block import TransformerClassifier
from config import argument_parser

def load_model(checkpoint_path, args):
    model = TransformerClassifier(args=args)
    if torch.cuda.is_available():
        model = model.cuda()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'state_dicts' in checkpoint:
        model.load_state_dict(checkpoint['state_dicts'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)   
    model.eval()
    return model

def load_dataset_info(dataset='PA100k'):
    with open(f'./data/{dataset}/dataset.pkl', 'rb') as f:
        dataset_info = pickle.load(f)
    return dataset_info

def preprocess_image(image_path, height=256, width=128):
    transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image

def visualize_attributes(image, predictions, attributes, threshold=0.5, output_path=None):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    y_offset = 10
    for i, (attr, pred) in enumerate(zip(attributes, predictions)):
        color = 'green' if pred > threshold else 'red'
        text = f"{attr}: {pred:.2f}"
        draw.text((10, y_offset), text, fill=color, font=font)
        y_offset += 25

    if output_path:
        image.save(output_path)
    return image

def visualize_summary(image, output_text, output_path=None):
    """
    Draws the specific summary string (Gender and Age) on the image.
    """
    draw = ImageDraw.Draw(image)
    try:
        # Increase font size for the summary
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
    except:
        font = ImageFont.load_default()

    # Split the string if it's too long, or just treat as one line
    # final_output_string is usually: "Gender: female, Age: age 18 to 60"
    
    # Optional: Draw a dark semi-transparent rectangle behind text for better visibility
    text_pos = (20, 20)
    
    draw.text(text_pos, output_text, fill='green', font=font)

    if output_path:
        image.save(output_path)
    return image

def main():
    parser = argparse.ArgumentParser(description="Inference for UniPAR")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--dataset', type=str, default='PA100k', help='Dataset name')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')

    args = parser.parse_args()

    # Load config args (for model initialization)
    config_args = argument_parser().parse_args([])  # Empty args for defaults
    config_args.dataset = [args.dataset]

    # Load model
    model = load_model(args.checkpoint, config_args)

    # Load dataset info
    dataset_info = load_dataset_info(args.dataset)
    attributes = dataset_info.attr_words
    word_vec = torch.tensor(dataset_info.attr_vectors, dtype=torch.float32)

    # Preprocess image
    input_tensor, original_image = preprocess_image(args.image)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        word_vec = word_vec.cuda()

    # Inference
    with torch.no_grad():
        logits = model(input_tensor, None, word_vec)
        predictions = torch.sigmoid(logits).cpu().numpy().flatten()

    # Visualize
    os.makedirs(args.output, exist_ok=True)
    output_filename = os.path.basename(args.image).replace('.jpg', '_pred.jpg').replace('.png', '_pred.png')
    output_path = os.path.join(args.output, output_filename)
    #visualize_attributes(original_image, predictions, attributes, args.threshold, output_path)

    # Wanted Label Text List from Attr:
    gender_attr = 'female'
    age_attrs = ['age over 60', 'age 18 to 60', 'age less 18']
    age_pred_list = []
    final_output_string = ""


    print(f"Inference completed. Results saved to {output_path}")
    print("Predicted attributes:")
    for attr, pred in zip(attributes, predictions):
        print(f"  {attr}: {pred:.3f}")
        if attr == gender_attr:
            attr = 'male' if pred <= 0.5 else 'female'
            final_output_string += f'{attr}, '   
        elif attr in age_attrs:
            age_pred_list.append(pred)
    if age_pred_list:
        age_index = np.argmax(age_pred_list) # -> Reutrn the index of the element
        final_output_string += f'{age_attrs[age_index]}'
    
    # New Visualization
    visualize_summary(original_image, final_output_string, output_path)
    print(f"Inference completed. Results saved to {output_path}")
    print(f'--- Considered Attribute ---\n{final_output_string}')
        
            

if __name__ == '__main__':
    main()