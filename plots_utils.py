import os
import matplotlib.pyplot as plt
import numpy as np

def plot_attention_vs_causation_scatter(scatter_rollout_hist, scatter_causal_hist, scatter_rollout_t0, scatter_causal_t0, save_dir):
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
    g1_path = os.path.join(save_dir, "graph1_attention_vs_causation_scatter.png")
    plt.savefig(g1_path, dpi=300)
    plt.close()
    print(f"Saved {g1_path}")


def plot_local_causal_tracing(avg_local_restoration, module_names, total_causal_items, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(module_names))
    
    ax.bar(x_pos, avg_local_restoration, color='purple', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(module_names, rotation=45, ha='right')
    ax.set_ylabel('Average Restoration Score')
    ax.set_title(f'Graph 2: Internal Processing (Local Token Patched, N={total_causal_items})')
    
    plt.tight_layout()
    g2_path = os.path.join(save_dir, "graph2_local_causal_tracing.png")
    plt.savefig(g2_path, dpi=300)
    plt.close()
    print(f"Saved {g2_path}")


def plot_position_distribution(causal_positions_count, total_causal_items, save_dir):
    if total_causal_items == 0:
        print("Not enough causal items for Graph 3.")
        return
        
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
    g3_path = os.path.join(save_dir, "graph3_position_distribution.png")
    plt.savefig(g3_path, dpi=300)
    plt.close()
    print(f"Saved {g3_path}")
