import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def plot_attention_vs_causation_scatter(scatter_rollout_hist, scatter_causal_hist, scatter_rollout_t0, scatter_causal_t0, save_dir):
    """
    Plots a scatter plot comparing Attention Rollout scores against Causal Impact scores.
    Historical tokens are shown in blue, while the last token (T-0) is highlighted in red.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot historical items
    ax.scatter(scatter_rollout_hist, scatter_causal_hist, color='blue', alpha=0.3, label='Historical Tokens (T-1, T-2...)', edgecolors='none')
    
    # Plot T-0
    ax.scatter(scatter_rollout_t0, scatter_causal_t0, color='red', alpha=0.8, marker='x', s=100, label='Last Token (T-0)')
    
    # Add a diagonal line for reference (perfect correlation)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Perfect Correlation (Attention = Causation)')
    
    ax.set_xlabel('Normalized Attention Rollout')
    ax.set_ylabel('Normalized Causal Impact (Logit Drop)')
    ax.set_title('Graph 1: Attention Rollout vs Causal Impact')
    ax.set_ylim(-2.5, 1.2)  # Zoom in to ignore extreme negative outliers
    ax.legend()
    
    plt.tight_layout()
    g1_path = os.path.join(save_dir, "graph1_attention_vs_causation_scatter.png")
    plt.savefig(g1_path, dpi=300)
    plt.close()
    print(f"Saved {g1_path}")


def plot_local_causal_tracing(avg_local_restoration, module_names, total_causal_items, save_dir):
    """
    Plots a bar chart showing the average restoration scores for each module when local tokens are patched.
    """
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
    """
    Plots a bar chart showing the distribution of highly causal items across different token positions.
    """
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


def plot_popularity_distribution(causal_popularity_count, total_causal_items, save_dir):
    """
    Plots a bar chart showing the distribution of highly causal items across different popularity deciles.
    """
    if total_causal_items == 0:
        print("Not enough causal items for Popularity Graph.")
        return
        
    # Standardize 10 deciles in order
    decile_labels = [f"Top {i*10}-{(i+1)*10}%" for i in range(10)]
    pos_pcts = []
    
    for label in decile_labels:
        count = causal_popularity_count.get(label, 0)
        pos_pcts.append((count / total_causal_items) * 100)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(decile_labels, pos_pcts, color='teal', alpha=0.8)
    
    ax.set_xlabel('Item Popularity Ranking (Deciles)')
    ax.set_ylabel('Percentage of Highly Causal Items (%)')
    ax.set_title(f'Graph 4: Popularity Distribution of Causal Items (N={total_causal_items})')
    ax.tick_params(axis='x', rotation=45)
    
    # Add exact percentages on top of the bars
    for i, v in enumerate(pos_pcts):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    g4_path = os.path.join(save_dir, "graph4_popularity_distribution.png")
    plt.savefig(g4_path, dpi=300)
    plt.close()
    print(f"Saved {g4_path}")


def plot_local_causal_tracing_box(graph2_data, module_names, total_causal_items, save_dir):
    """
    Plots a boxplot showing the distribution of restoration scores for each module when local tokens are patched.
    """
    if total_causal_items == 0:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [graph2_data[m] for m in module_names]
    
    # Create boxplot without showing extreme outliers to preserve the y-axis scale
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, showfliers=False)
    
    # Style it
    for box in bp['boxes']:
        box.set_facecolor('purple')
        box.set_alpha(0.6)
        
    ax.set_xticks(np.arange(1, len(module_names) + 1))
    ax.set_xticklabels(module_names, rotation=45, ha='right')
    ax.set_ylabel('Restoration Score')
    ax.set_title(f'Graph 2b: Internal Processing Boxplot (Local Token Patched, N={total_causal_items})')
    
    plt.tight_layout()
    g2b_path = os.path.join(save_dir, "graph2b_local_causal_tracing_box.png")
    plt.savefig(g2b_path, dpi=300)
    plt.close()
    print(f"Saved {g2b_path}")


def plot_position_distribution_box(causal_positions_raw, total_causal_items, save_dir):
    """
    Plots a boxplot showing the distribution of positions (T-N) of highly causal items.
    """
    if total_causal_items == 0 or len(causal_positions_raw) == 0:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([causal_positions_raw], patch_artist=True, vert=False, showmeans=True)
    
    for box in bp['boxes']:
        box.set_facecolor('orange')
        box.set_alpha(0.6)
        
    ax.set_yticks([])
    ax.set_xlabel('Relative Position (0 = T-0, 1 = T-1, etc.)')
    ax.set_title(f'Graph 3b: Position Distribution Boxplot (N={total_causal_items})')
    
    # Force x-axis to show only integers (no 0.5, 1.5 etc.)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    g3b_path = os.path.join(save_dir, "graph3b_position_distribution_box.png")
    plt.savefig(g3b_path, dpi=300)
    plt.close()
    print(f"Saved {g3b_path}")


def plot_popularity_distribution_box(causal_popularity_raw, total_causal_items, save_dir):
    """
    Plots a boxplot showing the distribution of popularity percentiles of highly causal items.
    """
    if total_causal_items == 0 or len(causal_popularity_raw) == 0:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([causal_popularity_raw], patch_artist=True, vert=False, showmeans=True)
    
    for box in bp['boxes']:
        box.set_facecolor('teal')
        box.set_alpha(0.6)
        
    ax.set_yticks([])
    ax.set_xlabel('Item Popularity Percentile (0% = Most Popular, 100% = Least Popular)')
    ax.set_title(f'Graph 4b: Popularity Percentile Boxplot (N={total_causal_items})')
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    g4b_path = os.path.join(save_dir, "graph4b_popularity_distribution_box.png")
    plt.savefig(g4b_path, dpi=300)
    plt.close()
    print(f"Saved {g4b_path}")
