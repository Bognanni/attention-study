import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_utils import get_test_dataloader
from utils import build_model, get_device, load_config

from attention_rollout import compute_attention_rollout
from causal_tracing import ActivationPatcher
from interpretability_eval import FaithfulnessEvaluator

def main():
    parser = argparse.ArgumentParser(description="Attention vs Causation")
    parser.add_argument('--config', type=str, default='config_ml1m_sasrec.py')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='protocol_plots')
    parser.add_argument('--num_samples', type=int, default=100, help="Number of sequences to process")
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

    # Data structures for the 3 final graphs
    scatter_rollout_hist, scatter_causal_hist = [], []
    scatter_rollout_t0, scatter_causal_t0 = [], []
    
    patcher = ActivationPatcher(model)
    module_names = list(patcher.modules.keys())
    graph2_data = {m: [] for m in module_names}
    
    causal_positions_count = {}
    total_causal_items = 0
    causal_head_count = 0

    # POPULARITY ANALYSIS SETUP
    train_file = f"datasets/{config.dataset_name}/train/input.txt"
    item_counts = {}
    total_train_interactions = 0
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            for line in f:
                items = line.strip().split()
                for item in items:
                    item_id = int(item)
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1
                    total_train_interactions += 1
                    
        # Sort items by popularity
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        num_unique_items = len(sorted_items)
        head_cutoff = int(num_unique_items * 0.20) # Top 20% are "Head" items
        head_items_set = set([item[0] for item in sorted_items[:head_cutoff]])
        
        # Calculate dataset-level popularity bias
        head_interactions = sum([item[1] for item in sorted_items[:head_cutoff]])
        dataset_head_ratio = head_interactions / total_train_interactions if total_train_interactions > 0 else 0
        print(f"\n--- POPULARITY BASELINE ---")
        print(f"Dataset Analysis: Top 20% of items (Head) account for {dataset_head_ratio*100:.2f}% of all historical interactions.")
    else:
        print(f"\nWarning: Train file not found at {train_file}. Skipping popularity baseline.")
        head_items_set = set()
        dataset_head_ratio = 0

    # Phase 1 Metrics Storage
    spearman_scores, jaccard_scores = [], []
    comp_scores, suff_scores = [], []
    causal_comp_scores, causal_suff_scores = [], []

    for b in tqdm(range(actual_samples), desc="Processing Sequences"):
        input_seq = input_seqs_batch[b]
        
        with torch.no_grad():
            base_logits = FaithfulnessEvaluator.get_logits(model, input_seq.unsqueeze(0))
            target_item = torch.argmax(base_logits, dim=-1).item()
            _, attentions = model(input_seq.unsqueeze(0))
            
        # PHASE 1: Ranking & Input Ablation
        rollout_matrix = compute_attention_rollout(attentions)
        rollout_scores = rollout_matrix[0, -1, :] # Score from target position
        
        causal_drops, is_highly_causal, valid_idx = FaithfulnessEvaluator.compute_loo_ablation(
            model, input_seq, target_item, pad_token_id
        )
        valid_idx_np = valid_idx.cpu().numpy()
        
        spearman = FaithfulnessEvaluator.compute_spearman(rollout_scores, causal_drops, valid_idx)
        jaccard = FaithfulnessEvaluator.compute_jaccard(rollout_scores, causal_drops, valid_idx, k=3)
        comp, suff = FaithfulnessEvaluator.compute_comprehensiveness_and_sufficiency(
            model, input_seq, target_item, pad_token_id, valid_idx, rollout_scores, k=3
        )
        causal_comp, causal_suff = FaithfulnessEvaluator.compute_comprehensiveness_and_sufficiency(
            model, input_seq, target_item, pad_token_id, valid_idx, causal_drops, k=3
        )
        
        spearman_scores.append(spearman)
        jaccard_scores.append(jaccard)
        comp_scores.append(comp)
        suff_scores.append(suff)
        causal_comp_scores.append(causal_comp)
        causal_suff_scores.append(causal_suff)
        
        # GRAPH 1 DATA: Attention vs Causation
        seq_max_rollout = rollout_scores[valid_idx].max().item()
        seq_max_causal = causal_drops[valid_idx].max().item()
        
        for idx in valid_idx_np:
            norm_r = rollout_scores[idx].item() / seq_max_rollout if seq_max_rollout > 0 else 0
            norm_c = causal_drops[idx].item() / seq_max_causal if seq_max_causal > 0 else 0
            
            # Separate T-0 from Historical Tokens
            if idx == valid_idx_np[-1]:
                scatter_rollout_t0.append(norm_r)
                scatter_causal_t0.append(norm_c)
            else:
                scatter_rollout_hist.append(norm_r)
                scatter_causal_hist.append(norm_c)
                
        # GRAPH 3 DATA: Positions of Highly Causal Items
        vital_indices = list(valid_idx_np[is_highly_causal[valid_idx_np].cpu().numpy()])
        for idx in vital_indices:
            actual_item_id = input_seq[idx].item()
            total_causal_items += 1
            if actual_item_id in head_items_set:
                causal_head_count += 1
            
            pos_in_valid = np.where(valid_idx_np == idx)[0][0]
            relative_pos = len(valid_idx_np) - 1 - pos_in_valid
            pos_label = f"T-{relative_pos}"
            causal_positions_count[pos_label] = causal_positions_count.get(pos_label, 0) + 1
            
        # GRAPH 2 DATA: Local Causal Tracing
        if len(vital_indices) == 0:
            continue
            
        results, _ = patcher.run_patching_protocol(
            input_seq, target_item, pad_token_id, vital_indices, valid_idx, epsilon=1e-3
        )
        
        for corrupt_pos in vital_indices:
            if corrupt_pos not in results:
                continue
            for m in module_names:
                # We record the Restoration Score patching the Local Token
                graph2_data[m].append(results[corrupt_pos][m][corrupt_pos].item())

    # TERMINAL STATS
    print("\n--- PHASE 1 DATASET RESULTS ---")
    print(f"Average Spearman Correlation: {np.mean(spearman_scores):.4f}")
    print(f"Average Top-3 Jaccard: {np.mean(jaccard_scores):.4f}")
    print(f"\n[Attention Rollout] Comp: {np.mean(comp_scores):.4f} | Suff: {np.mean(suff_scores):.4f}")
    print(f"[Causal Baseline]   Comp: {np.mean(causal_comp_scores):.4f} | Suff: {np.mean(causal_suff_scores):.4f}")

    print("\n--- CAUSAL POPULARITY RESULTS ---")
    if total_causal_items > 0:
        causal_head_ratio = causal_head_count / total_causal_items
        print(f"Popularity Amplification:")
        print(f"  - The model selected 'Head' (Top 20%) items as highly causal {causal_head_ratio*100:.2f}% of the time.")
        print(f"  - (Baseline was {dataset_head_ratio*100:.2f}% -> Amplification = {causal_head_ratio - dataset_head_ratio:+.2%})")

    print("\n--- GENERATING FINAL GRAPHS ---")
    
    # GRAPH 1: Attention vs Causation Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot historical items (The "False Positives" cloud)
    ax.scatter(scatter_rollout_hist, scatter_causal_hist, color='blue', alpha=0.3, label='Historical Tokens (T-1, T-2...)', edgecolors='none')
    
    # Plot T-0 (The "True Positives" cluster)
    ax.scatter(scatter_rollout_t0, scatter_causal_t0, color='red', alpha=0.8, marker='x', s=100, label='Last Token (T-0)')
    
    # Add a diagonal line for reference (perfect correlation)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Perfect Correlation (Attention = Causation)')
    
    ax.set_xlabel('Normalized Attention Rollout')
    ax.set_ylabel('Normalized Causal Impact (Logit Drop)')
    ax.set_title('Graph 1: Attention Rollout vs Causal Impact')
    ax.legend()
    
    plt.tight_layout()
    g1_path = os.path.join(args.save_dir, "graph1_attention_vs_causation_scatter.png")
    plt.savefig(g1_path, dpi=300)
    plt.close()
    print(f"Saved {g1_path}")

    # GRAPH 2: Causal Tracing (Local Token Processing)
    avg_local_restoration = [np.mean(graph2_data[m]) if graph2_data[m] else 0 for m in module_names]
    
    print("\n[Average Restoration Score by Module]")
    for i, m in enumerate(module_names):
        print(f"  - {m}: {avg_local_restoration[i]:.4f}")
        
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(module_names))
    
    ax.bar(x_pos, avg_local_restoration, color='purple', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(module_names, rotation=45, ha='right')
    ax.set_ylabel('Average Restoration Score')
    ax.set_title(f'Graph 2: Internal Processing (Local Token Patched, N={total_causal_items})')
    
    plt.tight_layout()
    g2_path = os.path.join(args.save_dir, "graph2_local_causal_tracing.png")
    plt.savefig(g2_path, dpi=300)
    plt.close()
    print(f"Saved {g2_path}")

    # GRAPH 3: Position Distribution (The importance of the last tokens)
    if total_causal_items > 0:
        sorted_positions = sorted(causal_positions_count.items(), key=lambda x: int(x[0].split('-')[1]))
        pos_labels = [p[0] for p in sorted_positions]
        pos_pcts = [(p[1] / total_causal_items) * 100 for p in sorted_positions]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(pos_labels, pos_pcts, color='orange', alpha=0.8)
        
        ax.set_xlabel('Token Position (T-0 = Last Token)')
        ax.set_ylabel('Percentage of Highly Causal Items (%)')
        ax.set_title(f'Graph 3: Position Distribution of Causal Items (N={total_causal_items})')
        
        # Add exact percentages on top of the bars
        for i, v in enumerate(pos_pcts):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
            
        plt.tight_layout()
        g3_path = os.path.join(args.save_dir, "graph3_position_distribution.png")
        plt.savefig(g3_path, dpi=300)
        plt.close()
        print(f"Saved {g3_path}")
    else:
        print("Not enough causal items for Graph 3.")

if __name__ == '__main__':
    main()
