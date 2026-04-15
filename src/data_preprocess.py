import torch
from collections import Counter
import pickle
import yaml
import os
from datasets import load_dataset

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_and_preprocess():
    print("Loading Tiny Shakespeare dataset...")
    
    # Important fix: trust_remote_code=True
    dataset = load_dataset(config["data"]["dataset"], 
                           split="train", 
                           trust_remote_code=True)
    
    text = dataset[0]['text'].lower()
    print(f"Downloaded {len(text):,} characters")
    
    # Word tokenization
    words = text.split()
    word_counts = Counter(words)
    
    # Build vocabulary
    vocab = ["<UNK>", "<PAD>"] + [w for w, c in word_counts.items() if c >= config["data"]["min_freq"]]
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Convert text to indices
    indices = [word_to_idx.get(word, 0) for word in words]
    
    seq_length = config["data"]["seq_length"]
    inputs, targets = [], []
    
    for i in range(len(indices) - seq_length):
        inputs.append(indices[i:i + seq_length])
        targets.append(indices[i + seq_length])
    
    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    # Train / Val split (90-10)
    split = int(0.9 * len(inputs))
    
    # Save tensors
    os.makedirs("data/processed", exist_ok=True)
    torch.save((inputs[:split], targets[:split], inputs[split:], targets[split:]), 
               "data/processed/data_tensors.pt")
    
    # Save vocab
    with open("data/processed/vocab.pkl", "wb") as f:
        pickle.dump((word_to_idx, idx_to_word, len(vocab)), f)
    
    print(f"✅ Vocabulary size: {len(vocab)}")
    print(f"✅ Total sequences created: {len(inputs)}")
    
    # Update config with vocab size
    config["data"]["vocab_size"] = len(vocab)
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    load_and_preprocess()