import os
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, f1_score

# Project imports
from config import argument_parser
from dataset.AttrDataset import get_multi_dataset
from models.base_block import TransformerClassifier
from tools.utils import set_seed, time_str

def load_dataset_info(dataset_name='PA100k'):
    # Adjust path if your pkl is stored elsewhere
    with open(f'./data/{dataset_name}/dataset.pkl', 'rb') as f:
        dataset_info = pickle.load(f)
    return dataset_info

def visualize_summary(image, prediction_data, output_path):
    """
    prediction_data: List of tuples -> [("text", "color"), ("text", "color")]
    """
    image = image.resize((256, 512), Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except:
        font = ImageFont.load_default()

    y_offset = 20
    for text, color in prediction_data:
        # Shadow for readability
        draw.text((12, y_offset + 2), text, fill='black', font=font)
        # Main text
        draw.text((10, y_offset), text, fill=color, font=font)
        y_offset += 30 # Move next line down

    image.save(output_path)

def main():
    set_seed(605)
    
    # 1. Setup
    parser = argument_parser()
    args = parser.parse_args()
    args.dataset = ['PA100k'] 
    
    # 2. Load Dataset
    # get_multi_dataset returns (train_set, valid_set, loss_dict)
    _, multi_valid_set, _ = get_multi_dataset(args)
    multi_valid_set.init_set('PA100k')
    
    # 3. Get Metadata (The fix for the AttributeError)
    dataset_info = load_dataset_info('PA100k')
    attributes = list(dataset_info.attr_words)
    word_vec = torch.tensor(dataset_info.attr_vectors, dtype=torch.float32)

    # Define indices for specific logic
    gender_idx = attributes.index('female')
    age_attrs = ['age over 60', 'age 18 to 60', 'age less 18']
    age_indices = [attributes.index(a) for a in age_attrs]

    # 4. Model Setup
    model = TransformerClassifier(args=args)
    checkpoint_path = "logs/pa100k_uniPAR_v2/2026-05-13_16_58_39/ckpt_2026-05-14_03_56_03_40.pth" 
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dicts' in checkpoint:
        state_dict = checkpoint['state_dicts']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    
    if torch.cuda.is_available():
        model = model.cuda()
        word_vec = word_vec.cuda()
    model.eval()

    # 5. Evaluation Loop
    all_gender_gt, all_gender_pred = [], []
    all_age_gt, all_age_pred = [], []
    
    output_dir = os.path.join('eval_results', time_str())
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    image_root = './data/PA100k/data/' 
    
    # 2. Get the list of filenames from the pkl you loaded earlier
    # PA100k pkl files usually store filenames in 'image_name'
    all_filenames = dataset_info.image_name
    test_indices = dataset_info.partition['test'] 
    
    # 2. Get ONLY the test filenames
    test_filenames = [dataset_info.image_name[idx] for idx in test_indices]

    print(f"Starting evaluation on {len(test_filenames)} test images...")
    # To decide Mean of the age baseline:
    sum = 0
    with torch.no_grad():
        for i in tqdm(range(len(multi_valid_set))):
            # Check the first few attributes to see the naming conventionye
            # print("--- Dataset Attribute Check ---")
            # for idx, name in enumerate(attributes[:5]):
            #     print(f"Index {idx}: {name}")
            # Accessing via index to get the raw image info
            img_tensor, gt_label, _, _ = multi_valid_set[i]
            
            # Prepare Input
            img_input = img_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                img_input = img_input.cuda()

            # Inference
            logits = model(img_input, None, word_vec)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Process Gender
            age_prob = probs[gender_idx]
            sum += age_prob
            pred_gender_idx = 1 if probs[gender_idx] > 0.5 else 0
            all_gender_gt.append(gt_label[gender_idx])
            all_gender_pred.append(pred_gender_idx)

            # Process Age (Winner-takes-all among the 3 age classes)
            p_age_idx = np.argmax([probs[idx] for idx in age_indices])
            g_age_idx = np.argmax([gt_label[idx] for idx in age_indices])
            all_age_gt.append(g_age_idx)
            all_age_pred.append(p_age_idx)
            # Get Ground Truth Strings for comparison
            gt_gender_idx = int(gt_label[gender_idx])
            gt_age_idx = np.argmax([gt_label[idx] for idx in age_indices])

            img_filename = test_filenames[i] 
            full_path = os.path.join(image_root, img_filename)
            
            if os.path.exists(full_path):
                raw_img = Image.open(full_path).convert('RGB')
                    
                    # 1. Determine Gender Color
                gender_str = 'female' if pred_gender_idx == 1 else 'male'
                gender_color = 'lime' if pred_gender_idx == gt_gender_idx else 'red'
                    
                    # 2. Determine Age Color
                age_str = age_attrs[p_age_idx]
                age_color = 'lime' if p_age_idx == gt_age_idx else 'red'
                    
                # 3. Create display list
                # We show "Pred (Truth)" if it's wrong to make it helpful
                gender_display = f"G: {gender_str}" if gender_color == 'lime' else f"G: {gender_str} (!)"
                age_display = f"A: {age_str}" if age_color == 'lime' else f"A: {age_str} (!)"
                    
                vis_data = [
                    (gender_display, gender_color),
                    (age_display, age_color)
                ]
                    
                visualize_summary(raw_img, vis_data, os.path.join(vis_dir, f"{i}.jpg"))

    # 6. Metrics Summary
    print("\n" + "="*40)
    print(f"FINAL METRICS - PA100k")
    print("-" * 40)
    print(f"Gender Accuracy: {accuracy_score(all_gender_gt, all_gender_pred):.4f}")
    print(f"Age Accuracy:    {accuracy_score(all_age_gt, all_age_pred):.4f}")
    print(f"Age Macro-F1:    {f1_score(all_age_gt, all_age_pred, average='macro'):.4f}")
    print("="*40)
    print(f"Visualizations saved to: {vis_dir}")
    print(f"Age Threshold: {sum/len(multi_valid_set)} in {len(multi_valid_set)}.")

if __name__ == '__main__':
    main()