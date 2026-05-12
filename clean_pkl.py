import pickle
import os

pkl_path = './data/PA100k/dataset.pkl'
img_dir = './data/PA100k/data/'

# 1. Get exact list of .jpg files (ignoring folders and hidden files)
actual_jpgs = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
print(f"Actual .jpg files on disk: {len(actual_jpgs)}")

# 2. Check the pickle
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
    pkl_names = data['image_name']
    print(f"Entries in pkl: {len(pkl_names)}")

# 3. Find the 'Ghost' images (In pkl but NOT on disk)
ghosts = [name for name in pkl_names if name not in actual_jpgs]

if ghosts:
    print(f"❌ Found {len(ghosts)} ghost images in your pkl that don't exist on disk!")
    print(f"Sample missing files: {ghosts[:5]}")
else:
    print("✅ No ghost images found in pkl.")

# 4. Check for 'Extra' images (On disk but NOT in pkl)
extras = [name for name in actual_jpgs if name not in pkl_names]
if extras:
    print(f"⚠️ Found {len(extras)} images on disk that are NOT in your pkl.")