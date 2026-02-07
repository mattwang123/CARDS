"""
Global Attribution Analysis
Identify which tokens trigger the 'Insufficient' classification.
"""
import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from collections import defaultdict

def run_global_analysis(args):
    # 1. Train Probe on the Fly
    print(f"Training probe on layer {args.layer}...")
    train_emb = np.load(args.train_embeddings)[:, args.layer, :]
    with open(args.train_data, 'r') as f:
        train_labels = [1 if x['is_sufficient'] else 0 for x in json.load(f)]
    
    probe = LogisticRegression(class_weight='balanced', C=1.0)
    probe.fit(train_emb, train_labels)
    
    # Extract the "Insufficiency Direction" (Vector pointing to class 0)
    # Logic: coef is (1, dim) pointing to class 1 (Sufficient).
    # So negative coef points to Insufficient.
    probe_vec = torch.tensor(-probe.coef_[0], device=args.device, dtype=torch.float32)
    print("✓ Probe vector extracted.")

    # 2. Load Model & Data
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, output_attentions=True).to(args.device)
    model.eval()

    with open(args.test_data, 'r') as f:
        data = json.load(f)
    # Only analyze INSUFFICIENT examples
    insufficient_data = [x for x in data if not x['is_sufficient']][:args.max_examples]

    # 3. Analyze Attention
    print(f"Analyzing {len(insufficient_data)} examples...")
    token_scores = defaultdict(list)

    for item in tqdm(insufficient_data):
        inputs = tokenizer(item['question'], return_tensors='pt').to(args.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get Last Layer Attention: (Batch, Heads, Seq, Seq)
            # Average over heads, look at Last Token's row
            attn = outputs.attentions[-1][0, :, -1, :].mean(dim=0) # (SeqLen,)
            
            # Optional: Weight by value projection (advanced)
            # For now, raw attention is a good proxy for "what did I look at?"
            
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        for tok, score in zip(tokens, attn):
            # Clean token
            clean_tok = tok.replace('Ġ', '').replace(' ', '').lower()
            if len(clean_tok) > 2: # Skip short words
                token_scores[clean_tok].append(score.item())

    # 4. Aggregate
    avg_scores = {k: sum(v)/len(v) for k, v in token_scores.items() if len(v) >= 5}
    sorted_tokens = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    print("\nTOP TRIGGER TOKENS (Highest Attention from Last Token):")
    for tok, score in sorted_tokens[:20]:
        print(f"{tok:<20} {score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--train_embeddings', required=True)
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--layer', type=int, default=18)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_examples', type=int, default=500)
    args = parser.parse_args()
    
    run_global_analysis(args)