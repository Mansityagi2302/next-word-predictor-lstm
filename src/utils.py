import torch
import pickle
from typing import Tuple

def load_vocab() -> Tuple[dict, dict, int]:
    """Loads the processed vocabulary mappings."""
    with open("data/processed/vocab.pkl", "rb") as f:
        word_to_idx, idx_to_word, vocab_size = pickle.load(f)
    return word_to_idx, idx_to_word, vocab_size

def calculate_perplexity(loss: float) -> float:
    """Computes perplexity from cross-entropy loss."""
    return torch.exp(torch.tensor(loss)).item()

def generate_text(model, word_to_idx, idx_to_word, prompt: str, 
                  max_new_tokens: int = 50, temperature: float = 1.0,
                  top_k: int = 5, top_p: float = 0.8, device="cpu"):
    model.eval()
    words = prompt.lower().split()
    input_idx = [word_to_idx.get(w, 0) for w in words]
    hidden = None
    generated = words[:]
    
    # NEW: Repetition Penalty factor (1.2 is a good start)
    repetition_penalty = 1.2

    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([input_idx[-30:]], dtype=torch.long).to(device)
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
        
        logits = output[0] / max(temperature, 1e-6)

        # Apply Repetition Penalty
        for word_id in set(input_idx[-10:]): # Look at the last 10 words
            if logits[word_id] < 0:
                logits[word_id] *= repetition_penalty
            else:
                logits[word_id] /= repetition_penalty

        probs = torch.softmax(logits, dim=-1)
        
        # Top-K
        topk_probs, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)))
        
        # Top-P (Nucleus)
        topk_probs_norm = topk_probs / topk_probs.sum()
        cum_probs = torch.cumsum(topk_probs_norm, dim=0)
        mask = cum_probs <= top_p
        mask[0] = True 
        
        final_probs = topk_probs_norm[mask] / topk_probs_norm[mask].sum()
        final_indices = topk_idx[mask]
        
        # Sample
        sample_idx = torch.multinomial(final_probs, 1).item()
        next_word_idx = final_indices[sample_idx].item()
        
        next_word = idx_to_word[next_word_idx]
        
        if next_word != "<UNK>":
            generated.append(next_word)
        
        input_idx.append(next_word_idx)
    
    return " ".join(generated)