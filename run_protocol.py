import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset_utils import get_test_dataloader
from utils import build_model, get_device, load_config

from attention_rollout import compute_attention_rollout
from causal_tracing import ActivationPatcher
from interpretability_eval import FaithfulnessEvaluator

def main():
    parser = argparse.ArgumentParser(description="Dataset-Level 3-Phase MI Protocol")
    parser.add_argument('--config', type=str, default='config_ml1m_sasrec.py')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='protocol_plots')
    parser.add_argument('--num_samples', type=int, default=100, help="Number of sequences to process")
    parser.add_argument('--history_len', type=int, default=20, help="Max recent history items to average for Strategy A")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    config = load_config(args.config)
    device = get_device()
    
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    pad_token_id = model.num_items + 1
    test_dataloader = get_test_dataloader(config.dataset_name, batch_size=args.num_samples, max_length=config.sequence_length)
    batch = next(iter(test_dataloader))
    input_seqs_batch = batch[0].to(device)
    actual_samples = input_seqs_batch.size(0)

    print(f"Loaded {actual_samples} sequences. Starting Protocol...")

    # Metrics for Phase 1 - Ranking
    spearman_scores, jaccard_scores = [], []
    comp_scores, suff_scores = [], []
    
    # Aggregators for Phase 3 - Strategy A (Positional Heatmap)
    patcher = ActivationPatcher(model)
    module_names = list(patcher.modules.keys())
    
    # [module_name, history_len]
    strategy_a_sums = {m: np.zeros(args.history_len) for m in module_names}
    strategy_a_counts = {m: np.zeros(args.history_len) for m in module_names}
    
    # Aggregators for Phase 3 - Strategy B (Highly Causal Items)
    # We will accumulate the raw rollout weight and MHA restoration score for vital items
    strategy_b_data = {'rollout': [], 'L0_MHA_restore': [], 'L1_MHA_restore': []}

    for b in tqdm(range(actual_samples), desc="Processing Sequences"):
        input_seq = input_seqs_batch[b]
        
        with torch.no_grad():
            base_logits = FaithfulnessEvaluator.get_logits(model, input_seq.unsqueeze(0))
            target_item = torch.argmax(base_logits, dim=-1).item()
            _, attentions = model(input_seq.unsqueeze(0))
            
        # PHASE 1: Ranking
        rollout_matrix = compute_attention_rollout(attentions)
        rollout_scores = rollout_matrix[0, -1, :] # Score from target position
        
        causal_drops, is_highly_causal, valid_idx = FaithfulnessEvaluator.compute_loo_ablation(
            model, input_seq, target_item, pad_token_id
        )
        
        spearman = FaithfulnessEvaluator.compute_spearman(rollout_scores, causal_drops, valid_idx)
        jaccard = FaithfulnessEvaluator.compute_jaccard(rollout_scores, causal_drops, valid_idx, k=3)
        comp, suff = FaithfulnessEvaluator.compute_comprehensiveness_and_sufficiency(
            model, input_seq, target_item, pad_token_id, valid_idx, rollout_scores, k=3
        )
        
        spearman_scores.append(spearman)
        jaccard_scores.append(jaccard)
        comp_scores.append(comp)
        suff_scores.append(suff)
        
        # PHASE 2: Causal Tracing
        # Note: We use an epsilon of 1e-3 on the logit drop to avoid math errors
        results, _, _ = patcher.run_patching_protocol(input_seq, target_item, pad_token_id, epsilon=1e-3)
        
        # PHASE 3: Data Aggregation
        # Align from the right (T-1, T-2, ..., T-history_len)
        valid_idx_np = valid_idx.cpu().numpy()
        for offset in range(1, args.history_len + 1):
            if offset <= len(valid_idx_np):
                pos = valid_idx_np[-offset]
                for m in module_names:
                    # Only add if the restoration score isn't 0.0 (skipped by filter)
                    if results[m][pos] != 0.0:
                        strategy_a_sums[m][offset-1] += results[m][pos].item()
                        strategy_a_counts[m][offset-1] += 1
                        
        # Strategy B Aggregation: Track items that caused >50% probability drop
        vital_indices = valid_idx_np[is_highly_causal[valid_idx_np].cpu().numpy()]
        for pos in vital_indices:
            # Normalize the rollout score for this sequence to [0,1] for fair visual comparison (divide by max rollout in this sequence so if the
            # value is 1.0, it means this item had the highest rollout weight in this sequence)
            seq_max_rollout = rollout_scores[valid_idx].max().item()
            norm_rollout = rollout_scores[pos].item() / seq_max_rollout if seq_max_rollout > 0 else 0
            
            strategy_b_data['rollout'].append(norm_rollout)
            strategy_b_data['L0_MHA_restore'].append(results['L0_MHA'][pos].item())
            if 'L1_MHA' in results:
                strategy_b_data['L1_MHA_restore'].append(results['L1_MHA'][pos].item())

    # --- PHASE 1 FINAL STATS ---
    print("\n--- PHASE 1 DATASET RESULTS ---")
    print(f"Average Spearman Correlation: {np.mean(spearman_scores):.4f}")
    print(f"Average Top-3 Jaccard: {np.mean(jaccard_scores):.4f}")
    print(f"Average Comprehensiveness (Drop): {np.mean(comp_scores):.4f}")
    print(f"Average Sufficiency (Retention): {np.mean(suff_scores):.4f}")

    # --- PHASE 3 PLOTTING ---
    print("\n--- PHASE 3 VISUALIZATION ---")
    
    # Strategy A: Positional Causal Heatmap
    heatmap_matrix = np.zeros((len(module_names), args.history_len))
    for i, m in enumerate(module_names):
        # Prevent divide by zero
        safe_counts = np.where(strategy_a_counts[m] == 0, 1, strategy_a_counts[m])
        heatmap_matrix[i, :] = strategy_a_sums[m] / safe_counts
        
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(heatmap_matrix, cmap="viridis", ax=ax, cbar=True, vmin=0.0, vmax=1.0,
                xticklabels=[f"T-{i}" for i in range(1, args.history_len + 1)],
                yticklabels=module_names)
    ax.set_title(f"Strategy A: Positional Causal Heatmap (Avg over {actual_samples} sequences)")
    ax.set_xlabel("Relative Historical Position")
    ax.set_ylabel("Architecture Module")
    plt.tight_layout()
    heatmap_path = os.path.join(args.save_dir, "strategy_a_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    # Strategy B: Debunking Grouped Bar Chart
    vital_count = len(strategy_b_data['rollout'])
    print(f"Found {vital_count} Highly Causal Items across the dataset.")
    
    if vital_count > 0:
        avg_rollout = np.mean(strategy_b_data['rollout'])
        avg_L0 = np.mean(strategy_b_data['L0_MHA_restore'])
        avg_L1 = np.mean(strategy_b_data['L1_MHA_restore']) if len(strategy_b_data['L1_MHA_restore']) > 0 else 0
        
        labels = ['L0_MHA', 'L1_MHA']
        rollout_means = [avg_rollout, avg_rollout]  # Baseline to compare against
        causal_means = [avg_L0, avg_L1]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x - width/2, rollout_means, width, label='Avg Normalized Attention Rollout', color='blue', alpha=0.7)
        ax.bar(x + width/2, causal_means, width, label='Avg Causal Restoration Score', color='red', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Strategy B: Debunking Attention (For Highly Causal Items)")
        ax.set_ylabel("Score [0.0 - 1.0]")
        ax.legend()
        
        plt.tight_layout()
        bar_path = os.path.join(args.save_dir, "strategy_b_barchart.png")
        plt.savefig(bar_path, dpi=300)
        plt.close()
    else:
        print("Warning: No highly causal items found (probability drop > 50%). Cannot render Strategy B chart.")

if __name__ == '__main__':
    main()
