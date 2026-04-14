"""
================================================================================
Latent Watchdog Calibration (Temporal Alignment Fix)
================================================================================
This script extracts the hidden states of a model at EXACTLY the N-th generated 
token (the "Writing" phase), trains a logistic regression probe, and outputs
a probability density plot to verify that the Q1 (Hallucination) and Q3 (Correct)
distributions are cleanly separated.
"""

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

def load_data(path, sample_size=400):
    """Loads a balanced subset of training data."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Assume data has 'is_sufficient' boolean and 'question' string
    suff = [d for d in data if d.get('is_sufficient', True)]
    insuff = [d for d in data if not d.get('is_sufficient', True)]
    
    # Balance the dataset
    min_len = min(len(suff), len(insuff), sample_size // 2)
    balanced_data = suff[:min_len] + insuff[:min_len]
    labels = [0] * min_len + [1] * min_len # 1 = Insufficient/Hallucination risk
    
    return balanced_data, labels

def extract_early_generation_states(model, tokenizer, texts, target_token_idx=4, layer=-1):
    """
    Forces the model to generate up to target_token_idx and extracts the hidden state.
    target_token_idx=4 means the 5th generated token.
    """
    hidden_states = []
    
    model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc=f"Extracting Token {target_token_idx+1} States"):
            # Format prompt (adjust if using specific chat templates)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            # Generate EXACTLY enough tokens to reach our target
            outputs = model.generate(
                **inputs,
                max_new_tokens=target_token_idx + 1,
                min_new_tokens=target_token_idx + 1,
                return_dict_in_generate=True,
                output_hidden_states=True,
                do_sample=False, # Greedy decoding for stable CoT start
                pad_token_id=tokenizer.eos_token_id
            )
            
            # outputs.hidden_states is a tuple of length (max_new_tokens).
            # We want the hidden states at our target generation step.
            # Shape: tuple of layers -> tensor (batch=1, seq_len=1, hidden_dim)
            step_states = outputs.hidden_states[target_token_idx]
            
            # Extract the specific layer (usually the last or second-to-last)
            target_layer_state = step_states[layer][0, -1, :].cpu().numpy()
            hidden_states.append(target_layer_state)
            
    return np.array(hidden_states)

def main():
    # 1. Configuration
    model_id = "Qwen/Qwen2.5-Math-7B-Instruct" 
    
    data_path = "src/data/processed/insufficient_dataset_umwp/umwp_train.json"
    target_token = 12 # 0-indexed. 4 = the 5th generated token.
    target_layer = -4 # Last layer. (Try -2 or -3 if last layer is collapsed)
    
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16
    )

    # 2. Load and extract data
    print("Loading training subset...")
    data, labels = load_data(data_path, sample_size=600)
    questions = [d['question'] for d in data] # Adjust key based on your JSON schema
    
    X = extract_early_generation_states(model, tokenizer, questions, target_token_idx=target_token, layer=target_layer)
    y = np.array(labels)
    
    # 3. Train/Test Split & Train Watchdog
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("\nTraining Watchdog Probe (Logistic Regression)...")
    probe = LogisticRegression(
        max_iter=2000, 
        class_weight='balanced',
        C=0.01,           # Strong L2 penalty to force it to find a universal feature
        solver='liblinear' # Better solver for high-dim, low-sample data
    )
    probe.fit(X_train, y_train)
    
    train_acc = probe.score(X_train, y_train)
    test_acc = probe.score(X_test, y_test)
    print(f"Train Accuracy: {train_acc:.3f} | Test Accuracy: {test_acc:.3f}")
    
    # 4. Generate the Density Plot (The Sanity Check)
    probs = probe.predict_proba(X_test)[:, 1] # Prob of being "Insufficient"
    
    probs_suff = probs[y_test == 0] # Analogous to Q3
    probs_insuff = probs[y_test == 1] # Analogous to Q1/Q4
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.kdeplot(probs_suff, fill=True, color="blue", label="Sufficient (Valid CoT Start)")
    sns.kdeplot(probs_insuff, fill=True, color="red", label="Insufficient (Hallucination Risk)")
    
    plt.title(f"Latent Watchdog: Token {target_token+1} State Separation\nModel: {model_id.split('/')[-1]} | Layer: {target_layer}", fontweight='bold')
    plt.xlabel("Probability of Insufficiency")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    
    out_img = f"experiments/watchdog_calibration_{model_id.split('/')[-1]}.png"
    os.makedirs("experiments", exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"\n[SUCCESS] Calibration plot saved to: {out_img}")

if __name__ == "__main__":
    main()