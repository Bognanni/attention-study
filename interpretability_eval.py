import argparse
import os
import torch
import numpy as np
from scipy.stats import spearmanr
from typing import Callable
from tqdm import tqdm

from dataset_utils import get_test_dataloader
from utils import build_model, get_device, load_config
from attention_rollout import compute_attention_rollout
from attention_flow import compute_attention_flow
from causal_tracing import get_logits, get_ablation_hook

# ==========================================
# EVALUATION METRICS
# ==========================================

def get_valid_indices(input_sequence: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Helper to extract indices of valid, non-padding items.
    """
    return (input_sequence != pad_token_id).nonzero(as_tuple=True)[0]

def compute_spearman_correlation(rollout_scores: torch.Tensor, causal_scores: torch.Tensor, input_sequence: torch.Tensor, 
                                 pad_token_id: int) -> float:
    """
    Computes Spearman's Rank Correlation between rollout and causal scores,
    explicitly ignoring padding tokens.
    """
    valid_idx = get_valid_indices(input_sequence, pad_token_id)
    if len(valid_idx) < 2:
        return 0.0
        
    r_scores = rollout_scores[valid_idx].detach().cpu().numpy()
    c_scores = causal_scores[valid_idx].detach().cpu().numpy()
    
    correlation, _ = spearmanr(r_scores, c_scores)
    return float(correlation) if not np.isnan(correlation) else 0.0

def compute_topk_jaccard(rollout_scores: torch.Tensor, causal_scores: torch.Tensor, input_sequence: torch.Tensor,
                         pad_token_id: int, k: int = 3) -> float:
    """
    Computes Jaccard Similarity of the Top-K identified items between both methods.
    """
    valid_idx = get_valid_indices(input_sequence, pad_token_id)
    if len(valid_idx) == 0:
        return 0.0
        
    r_scores = rollout_scores[valid_idx]
    c_scores = causal_scores[valid_idx]
    
    actual_k = min(k, len(valid_idx))
    
    topk_r_local = torch.topk(r_scores, actual_k).indices
    topk_c_local = torch.topk(c_scores, actual_k).indices
    
    topk_r = set(valid_idx[topk_r_local].tolist())
    topk_c = set(valid_idx[topk_c_local].tolist())
    
    intersection = len(topk_r.intersection(topk_c))
    union = len(topk_r.union(topk_c))
    
    return intersection / union if union > 0 else 0.0

def compute_comprehensiveness(importance_scores: torch.Tensor, input_sequence: torch.Tensor, model: torch.nn.Module,
                              get_logits_fn: Callable, target_item: int, pad_token_id: int, k: int = 3) -> float:
    """
    Comprehensiveness (AOPC): Masks the Top-K most important items.
    A higher drop implies the method correctly identified critical items.
    """
    seq_2d = input_sequence.unsqueeze(0) if input_sequence.dim() == 1 else input_sequence
    with torch.no_grad():
        base_logits = get_logits_fn(model, seq_2d)
        base_logit = base_logits[0, target_item].item()
        
    valid_idx = get_valid_indices(input_sequence, pad_token_id)
    actual_k = min(k, len(valid_idx))
    
    if actual_k == 0:
        return 0.0
        
    topk_local = torch.topk(importance_scores[valid_idx], actual_k).indices
    topk_global = valid_idx[topk_local]
    
    masked_seq = input_sequence.clone()
    masked_seq[topk_global] = pad_token_id
    
    masked_seq_2d = masked_seq.unsqueeze(0) if masked_seq.dim() == 1 else masked_seq
    with torch.no_grad():
        masked_logits = get_logits_fn(model, masked_seq_2d)
        masked_logit = masked_logits[0, target_item].item()
        
    return base_logit - masked_logit

def compute_sufficiency(importance_scores: torch.Tensor, input_sequence: torch.Tensor, model: torch.nn.Module,
                        get_logits_fn: Callable, target_item: int, pad_token_id: int, k: int = 3) -> float:
    """
    Sufficiency: Masks ALL valid items EXCEPT the Top-K.
    A drop close to 0 implies the Top-K items alone are sufficient to retain the prediction.
    """
    seq_2d = input_sequence.unsqueeze(0) if input_sequence.dim() == 1 else input_sequence
    with torch.no_grad():
        base_logits = get_logits_fn(model, seq_2d)
        base_logit = base_logits[0, target_item].item()
        
    valid_idx = get_valid_indices(input_sequence, pad_token_id)
    actual_k = min(k, len(valid_idx))
    
    if actual_k == 0:
        return 0.0
        
    topk_local = torch.topk(importance_scores[valid_idx], actual_k).indices
    topk_global = valid_idx[topk_local]
    
    mask_to_keep = torch.zeros_like(input_sequence, dtype=torch.bool)
    mask_to_keep[topk_global] = True
    
    masked_seq = input_sequence.clone()
    for idx in valid_idx:
        if not mask_to_keep[idx]:
            masked_seq[idx] = pad_token_id
            
    masked_seq_2d = masked_seq.unsqueeze(0) if masked_seq.dim() == 1 else masked_seq
    with torch.no_grad():
        masked_logits = get_logits_fn(model, masked_seq_2d)
        masked_logit = masked_logits[0, target_item].item()
        
    return base_logit - masked_logit

def compute_incremental_aopc(
    importance_scores: torch.Tensor,
    input_sequence: torch.Tensor,
    model: torch.nn.Module,
    get_logits_fn: Callable,
    target_item: int,
    pad_token_id: int,
    k: int = 3
) -> float:
    """
    Incremental AOPC: Iteratively masks the top 1 to K most important items.
    Calculates the logit drop at each step, and returns the mean drop (Area Over Perturbation Curve).
    """
    seq_2d = input_sequence.unsqueeze(0) if input_sequence.dim() == 1 else input_sequence
    with torch.no_grad():
        base_logits = get_logits_fn(model, seq_2d)
        base_logit = base_logits[0, target_item].item()
        
    valid_idx = get_valid_indices(input_sequence, pad_token_id)
    actual_k = min(k, len(valid_idx))
    
    if actual_k == 0:
        return 0.0
        
    # Sort valid indices by importance descending
    sorted_local = torch.argsort(importance_scores[valid_idx], descending=True)
    sorted_global = valid_idx[sorted_local]
    
    drops = []
    masked_seq = input_sequence.clone()
    
    for step in range(actual_k):
        # Mask the (step)-th most important item
        item_to_mask = sorted_global[step]
        masked_seq[item_to_mask] = pad_token_id
        
        masked_seq_2d = masked_seq.unsqueeze(0) if masked_seq.dim() == 1 else masked_seq
        with torch.no_grad():
            masked_logits = get_logits_fn(model, masked_seq_2d)
            masked_logit = masked_logits[0, target_item].item()
            
        drop = base_logit - masked_logit
        drops.append(drop)
        
    # AOPC is the mean of drops over the K steps
    return float(np.mean(drops))


# ==========================================
# BATCH EXECUTION & TESTING
# ==========================================

def compute_batch_causal_scores(model, input_seqs, target_items, target_layer=0, patch_target='layer'):
    """
    Efficiently computes the causal logit drops for an entire batch at once.
    Allows targeting specific submodules (layer, mha, ffn).
    """
    batch_size = input_seqs.size(0)
    seq_len = input_seqs.size(1)
    
    causal_scores = torch.zeros((batch_size, seq_len), device=input_seqs.device)
    
    with torch.no_grad():
        base_logits = get_logits(model, input_seqs)
        base_logits_vals = base_logits[torch.arange(batch_size), target_items]
        
    # Dynamically select the target module based on user input
    if patch_target == 'layer':
        target_module = model.transformer_blocks[target_layer]
    elif patch_target == 'mha':
        target_module = model.transformer_blocks[target_layer].multihead_attention
    elif patch_target == 'ffn':
        target_module = model.transformer_blocks[target_layer].dense2
    else:
        raise ValueError(f"Unknown patch target: {patch_target}")
    
    for p in tqdm(range(seq_len), desc=f"Computing Causal Drops (Layer {target_layer} | {patch_target.upper()})"):
        hook_handle = target_module.register_forward_hook(get_ablation_hook(target_position=p))
        try:
            with torch.no_grad():
                patched_logits = get_logits(model, input_seqs)
                patched_logits_vals = patched_logits[torch.arange(batch_size), target_items]
                
                drop = base_logits_vals - patched_logits_vals
                causal_scores[:, p] = drop
        finally:
            hook_handle.remove()
            
    return causal_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate Interpretability Methods")
    parser.add_argument('--config', type=str, default='config_ml1m_sasrec.py')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=10, help="Number of sequences to test")
    parser.add_argument('--k', type=int, default=3, help="K for Top-K metrics")
    parser.add_argument('--layer', type=int, default=0, help='Layer to ablate for causal tracing')
    parser.add_argument('--patch_target', type=str, choices=['layer', 'mha', 'ffn'], default='layer',
                        help='Which submodule to ablate for causal tracing')
    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    device = get_device()
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    pad_token_id = model.num_items + 1

    test_dataloader = get_test_dataloader(config.dataset_name, batch_size=args.num_samples, max_length=config.sequence_length)
    batch = next(iter(test_dataloader))
    input_seqs = batch[0].to(device)
    
    print(f"Testing on batch size: {input_seqs.size(0)}")
    
    # Baseline
    with torch.no_grad():
        base_logits = get_logits(model, input_seqs)
        target_items = torch.argmax(base_logits, dim=-1)
        _, attentions = model(input_seqs)
        
    print("Computing Attention Rollout...")
    rollout_matrix = compute_attention_rollout(attentions)
    
    print("Computing Attention Flow...")
    # compute_attention_flow returns a numpy array, convert to tensor
    flow_matrix_np = compute_attention_flow(attentions)
    flow_matrix = torch.tensor(flow_matrix_np, device=device)
    
    print("Computing Causal Drops (Zero-Ablation)...")
    causal_matrix = compute_batch_causal_scores(model, input_seqs, target_items, target_layer=args.layer, patch_target=args.patch_target)
    
    print(f"\n--- Running Evaluations (K={args.k}) ---")
    metrics = {
        "rollout_spearman": [], "rollout_jaccard": [], 
        "rollout_comp": [], "rollout_suff": [], "rollout_aopc": [],
        "flow_spearman": [], "flow_jaccard": [], 
        "flow_comp": [], "flow_suff": [], "flow_aopc": [],
        "causal_comp": [], "causal_suff": [], "causal_aopc": []
    }
    
    for b in range(input_seqs.size(0)):
        seq = input_seqs[b]
        target = target_items[b].item()
        
        # Scores: attention from the last token to all other tokens
        r_scores = rollout_matrix[b, -1, :]
        f_scores = flow_matrix[b, -1, :]
        c_scores = causal_matrix[b, :]
        
        # Spearman & Jaccard (Rollout vs Causal)
        metrics["rollout_spearman"].append(compute_spearman_correlation(r_scores, c_scores, seq, pad_token_id))
        metrics["rollout_jaccard"].append(compute_topk_jaccard(r_scores, c_scores, seq, pad_token_id, k=args.k))
        
        # Spearman & Jaccard (Flow vs Causal)
        metrics["flow_spearman"].append(compute_spearman_correlation(f_scores, c_scores, seq, pad_token_id))
        metrics["flow_jaccard"].append(compute_topk_jaccard(f_scores, c_scores, seq, pad_token_id, k=args.k))
        
        # Comprehensiveness, Sufficiency, & Incremental AOPC (Rollout)
        metrics["rollout_comp"].append(compute_comprehensiveness(r_scores, seq, model, get_logits, target, pad_token_id, k=args.k))
        metrics["rollout_suff"].append(compute_sufficiency(r_scores, seq, model, get_logits, target, pad_token_id, k=args.k))
        metrics["rollout_aopc"].append(compute_incremental_aopc(r_scores, seq, model, get_logits, target, pad_token_id, k=args.k))
        
        # Comprehensiveness, Sufficiency, & Incremental AOPC (Flow)
        metrics["flow_comp"].append(compute_comprehensiveness(f_scores, seq, model, get_logits, target, pad_token_id, k=args.k))
        metrics["flow_suff"].append(compute_sufficiency(f_scores, seq, model, get_logits, target, pad_token_id, k=args.k))
        metrics["flow_aopc"].append(compute_incremental_aopc(f_scores, seq, model, get_logits, target, pad_token_id, k=args.k))
        
        # Comprehensiveness, Sufficiency, & Incremental AOPC (Causal/Oracle)
        metrics["causal_comp"].append(compute_comprehensiveness(c_scores, seq, model, get_logits, target, pad_token_id, k=args.k))
        metrics["causal_suff"].append(compute_sufficiency(c_scores, seq, model, get_logits, target, pad_token_id, k=args.k))
        metrics["causal_aopc"].append(compute_incremental_aopc(c_scores, seq, model, get_logits, target, pad_token_id, k=args.k))
        
    print(f"\n==========================================")
    print(f"RESULTS OVER {input_seqs.size(0)} SEQUENCES (K={args.k})")
    print(f"==========================================")
    
    print("\nAttention Rollout vs Causal Tracing:")
    print(f"  Spearman Correlation:     {np.mean(metrics['rollout_spearman']):.4f}")
    print(f"  Top-{args.k} Jaccard Similarity:  {np.mean(metrics['rollout_jaccard']):.4f}")
    
    print("\nAttention Flow vs Causal Tracing:")
    print(f"  Spearman Correlation:     {np.mean(metrics['flow_spearman']):.4f}")
    print(f"  Top-{args.k} Jaccard Similarity:  {np.mean(metrics['flow_jaccard']):.4f}")
    
    print("\n--- Model Degradation Metrics ---")
    print("\nAttention Rollout Performance:")
    print(f"  Fixed-K Comprehensiveness: {np.mean(metrics['rollout_comp']):.4f} (Higher is better)")
    print(f"  Incremental AOPC:          {np.mean(metrics['rollout_aopc']):.4f} (Higher is better)")
    print(f"  Sufficiency Drop:          {np.mean(metrics['rollout_suff']):.4f} (Lower is better)")
    
    print("\nAttention Flow Performance:")
    print(f"  Fixed-K Comprehensiveness: {np.mean(metrics['flow_comp']):.4f} (Higher is better)")
    print(f"  Incremental AOPC:          {np.mean(metrics['flow_aopc']):.4f} (Higher is better)")
    print(f"  Sufficiency Drop:          {np.mean(metrics['flow_suff']):.4f} (Lower is better)")
    
    print("\nCausal Tracing Performance:")
    print(f"  Fixed-K Comprehensiveness: {np.mean(metrics['causal_comp']):.4f} (Higher is better)")
    print(f"  Incremental AOPC:          {np.mean(metrics['causal_aopc']):.4f} (Higher is better)")
    print(f"  Sufficiency Drop:          {np.mean(metrics['causal_suff']):.4f} (Lower is better)")
    
if __name__ == '__main__':
    main()
