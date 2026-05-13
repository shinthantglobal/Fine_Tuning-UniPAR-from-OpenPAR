import pickle
import pandas as pd
import numpy as np

def export_ground_truth(dataset_name='PA100k', output_file='pa100k_test_ground_truth.csv'):
    # 1. Load the pickle
    with open(f'./data/{dataset_name}/dataset.pkl', 'rb') as f:
        dataset_info = pickle.load(f)
    
    # 2. Extract components
    # PA100k structure: image_name (list of strings), label (numpy array of 0s and 1s)
    image_names = dataset_info.image_name
    labels = dataset_info.label
    attr_names = dataset_info.attr_words
    
    # 3. Create a DataFrame
    # This maps each image to its 26 attributes
    df = pd.DataFrame(labels, columns=attr_names)
    
    # Insert the image name as the first column
    df.insert(0, 'image_id', image_names)
    
    # 4. Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Ground truth exported to {output_file}")
    print(f"Total images: {len(df)}")

if __name__ == '__main__':
    export_ground_truth()