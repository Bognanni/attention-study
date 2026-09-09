import argparse
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dataset_utils import get_test_dataloader
from utils import build_model, get_device, load_config

def plot_attention_heatmaps(attentions, item_ids, save_dir="attention_plots", seq_idx=0):
    """
    Plots attention heatmaps for a given sequence from the attention weights with Item IDs.
    
    Args:
        attentions: List of attention tensors from each transformer block.
                    Shape of each tensor: [batch_size, num_heads, seq_len, seq_len]
        item_ids: The item IDs for the specific sequence.
        save_dir: Directory to save the plots.
        seq_idx: Index of the sequence in the batch to visualize.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_blocks = len(attentions)
    num_heads = attentions[0].size(1)
    
    # Create a figure with subplots: num_blocks rows, num_heads columns
    # We increase the base figsize to fit the 200 labels
    fig, axes = plt.subplots(num_blocks, num_heads, figsize=(20 * num_heads, 16 * num_blocks), squeeze=False)
    
    # Convert item IDs to strings for labels
    labels = [str(int(x)) for x in item_ids]
    
    for block_idx in range(num_blocks):
        # Extract attention weights for the specific sequence
        # Shape: [num_heads, seq_len, seq_len]
        attn = attentions[block_idx][seq_idx].detach().cpu().numpy()
        
        for head_idx in range(num_heads):
            ax = axes[block_idx, head_idx]
            # Plot the heatmap for the specific head
            sns.heatmap(attn[head_idx], cmap="viridis", ax=ax, cbar=True, vmin=0, vmax=1,
                        xticklabels=labels, yticklabels=labels)
            ax.set_title(f"Block {block_idx + 1}, Head {head_idx + 1}")
            ax.set_ylabel("Target (Query) Item ID")
            ax.set_xlabel("Source (Key) Item ID")
            
            ax.tick_params(axis='x', labelsize=6, rotation=90)
            ax.tick_params(axis='y', labelsize=6, rotation=0)
            
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"attention_heatmap_seq_{seq_idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Attention heatmap saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract and visualize attention weights")
    parser.add_argument('--config', type=str, default='config_ml1m_sasrec.py', help='Configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--save_dir', type=str, default='attention_plots', help='Directory to save the heatmaps')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of sequences to visualize from the test set')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Build model
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Get data
    print("Loading test dataloader...")
    test_dataloader = get_test_dataloader(config.dataset_name, batch_size=args.num_samples, max_length=config.sequence_length)
    
    # Get a single batch
    batch = next(iter(test_dataloader))
    input_seqs = batch[0].to(device)
    
    print(f"Processing batch of shape: {input_seqs.shape}")
    
    # Forward pass to get attention weights
    with torch.no_grad():
        _, attentions = model(input_seqs)
        
    print(f"Extracted attention from {len(attentions)} transformer blocks.")
    
    # Plot and save heatmaps
    print("Generating heatmaps...")
    for i in range(min(args.num_samples, input_seqs.size(0))):
        plot_attention_heatmaps(attentions, input_seqs[i], save_dir=args.save_dir, seq_idx=i)
        
if __name__ == '__main__':
    main()
