import os
import random
import shutil

def select_reproducible_images(source_dir, dest_dir, sample_size=200, seed_value=42):
    # 1. Set the seed for reproducibility
    random.seed(seed_value)
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # 2. IMPORTANT: Sort the list first
    # os.listdir order can vary by OS; sorting ensures the starting point is identical
    images = sorted([f for f in os.listdir(source_dir) 
                     if f.lower().endswith(valid_extensions)])

    if len(images) < sample_size:
        sample_size = len(images)

    # 3. This will now pick the same images every time the script runs
    selection = random.sample(images, sample_size)

    for image_name in selection:
        shutil.copy2(os.path.join(source_dir, image_name), 
                     os.path.join(dest_dir, image_name))
        
    print(f"Reproducibly copied {sample_size} images using seed {seed_value}")

# --- Configuration ---
SOURCE = './eval_results/2026-05-13_11_13_38/visualizations'
DEST = 'Visuals/UniPAR_epoch_5'


os.makedirs(DEST, exist_ok=True)
# Use the same seed across all your evaluation runs
select_reproducible_images(SOURCE, DEST, seed_value=42)