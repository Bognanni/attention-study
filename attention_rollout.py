import argparse
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_utils import get_test_dataloader
from utils import build_model, get_device, load_config

def compute_attention_rollout(attentions):
    """
    Computes the Attention Rollout following the paper:
    "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, ACL 2020).
    
    Args:
        attentions: List of attention tensors from each transformer block.
                    Shape of each tensor: [batch_size, num_heads, seq_len, seq_len]
                    
    Returns:
        rollout: The computed attention rollout matrix of shape [batch_size, seq_len, seq_len]
    """
    batch_size = attentions[0].size(0)
    seq_len = attentions[0].size(2)
    
    # Initialize rollout as the identity matrix (representing the initial state)
    # Shape: [batch_size, seq_len, seq_len]
    rollout = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1).to(attentions[0].device)
    
    for layer_attention in attentions:
        # Average attention weights across all heads for this layer
        # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
        avg_attention = layer_attention.mean(dim=1) # Shape: [batch_size, seq_len, seq_len]
        
        # Add Identity matrix to account for residual connections
        # Transformer layer is essentially: output = output + attention(output)
        identity = torch.eye(seq_len).unsqueeze(0).to(layer_attention.device)
        attn_with_residual = avg_attention + identity
        
        # Row-normalize the matrix so that each row sums to 1 (making it stochastic)
        row_sums = attn_with_residual.sum(dim=-1, keepdim=True)
        # Avoid division by zero
        row_sums = torch.clamp(row_sums, min=1e-9)
        norm_attention = attn_with_residual / row_sums
        
        # Multiply with the rollout from previous layers
        # Matrix multiplication tracks how attention flows through the layers
        rollout = torch.bmm(norm_attention, rollout)
        
    return rollout

def plot_rollout_heatmaps(rollout, item_ids, save_dir="attention_plots", seq_idx=0):
    """
    Plots the Attention Rollout matrix as a heatmap with Item IDs on the axes.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Extract the rollout for the specific sequence and move to CPU
    r_matrix = rollout[seq_idx].detach().cpu().numpy()
    
    # Convert item IDs to strings for labels
    labels = [str(int(x)) for x in item_ids]
    
    sns.heatmap(r_matrix, cmap="viridis", ax=ax, cbar=True, vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels)
    ax.set_title("Attention Rollout (Information Flow)")
    ax.set_ylabel("Target (Query) Item ID")
    ax.set_xlabel("Source (Key) Item ID")
    
    ax.tick_params(axis='x', labelsize=6, rotation=90)
    ax.tick_params(axis='y', labelsize=6, rotation=0)
            
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"attention_rollout_seq_{seq_idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Attention Rollout heatmap saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract and visualize Attention Rollout")
    parser.add_argument('--config', type=str, default='config_ml1m_sasrec.py', help='Configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--save_dir', type=str, default='attention_plots', help='Directory to save the heatmaps')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of sequences to visualize from the test set')
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
    
    # Compute Rollout
    print("Computing Attention Rollout...")
    rollout = compute_attention_rollout(attentions)
    
    # Plot and save heatmaps
    print("Generating Rollout heatmaps...")
    for i in range(min(args.num_samples, input_seqs.size(0))):
        plot_rollout_heatmaps(rollout, input_seqs[i], save_dir=args.save_dir, seq_idx=i)
        
if __name__ == '__main__':
    main()
