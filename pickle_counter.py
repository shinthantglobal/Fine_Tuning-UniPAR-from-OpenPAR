import pickle
with open('./data/PA100k/dataset.pkl', 'rb') as f:
    data = pickle.load(f)
    print("Partitions:", data.partition.keys())
    for k in data.partition.keys():
        print(f"Key: {k}, Count: {len(data.partition[k])}")