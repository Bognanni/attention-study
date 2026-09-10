import argparse
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from tqdm import tqdm

from dataset_utils import get_test_dataloader
from utils import build_model, get_device, load_config

from multiprocessing import Pool, cpu_count
from functools import partial

def _compute_single_target_flow(i, seq_len, num_layers, G):
    """Helper function to run max-flow for one row (target node) to allow multiprocessing."""
    row = np.zeros(seq_len)
    target_node = f"L{num_layers}_{i}"
    
    if not G.has_node(target_node):
        return i, row
        
    for j in range(seq_len):
        source_node = f"L0_{j}"
        if G.has_node(source_node) and nx.has_path(G, source_node, target_node):
            row[j] = nx.maximum_flow_value(G, source_node, target_node)
    return i, row

def compute_attention_flow(attentions):
    """
    Computes Attention Flow using the max-flow algorithm from 
    "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, ACL 2020).
    """
    batch_size = attentions[0].size(0)
    seq_len = attentions[0].size(2)
    num_layers = len(attentions)
    
    flow_matrices = []
    
    # Check CPU cores for multiprocessing
    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores for parallel Max-Flow computation...")
    
    for b in range(batch_size):
        print(f"Computing Attention Flow for sequence {b+1}/{batch_size}...")
        
        # 1. Compute normalized attention matrices for all layers
        norm_attns = []
        for l in range(num_layers):
            layer_attn = attentions[l][b].mean(dim=0)
            identity = torch.eye(seq_len).to(layer_attn.device)
            attn_with_res = layer_attn + identity
            
            row_sums = attn_with_res.sum(dim=-1, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1e-9)
            norm_attn = attn_with_res / row_sums
            norm_attns.append(norm_attn.detach().cpu().numpy())
            
        # 2. Build the Directed Acyclic Graph (DAG) for Max-Flow
        print("Building flow network graph...")
        G = nx.DiGraph()
        for l in range(num_layers):
            A = norm_attns[l]
            for i in range(seq_len):
                for j in range(seq_len):
                    if A[i, j] > 1e-5: 
                        u = f"L{l}_{j}"
                        v = f"L{l+1}_{i}"
                        G.add_edge(u, v, capacity=A[i, j])
                        
        # 3. Compute Max Flow for all pairs using Multiprocessing
        flow_matrix = np.zeros((seq_len, seq_len))
        
        func = partial(_compute_single_target_flow, seq_len=seq_len, num_layers=num_layers, G=G)
        
        with Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap(func, range(seq_len)), total=seq_len, desc="Calculating Max Flow"))
            
        for i, row in results:
            flow_matrix[i, :] = row
                    
        flow_matrices.append(flow_matrix)
        
    return np.array(flow_matrices)

def compute_per_head_attention_flow(attentions):
    """
    Computes Attention Flow independently for each head.
    Returns: shape [batch_size, num_heads, seq_len, seq_len]
    """
    batch_size = attentions[0].size(0)
    num_heads = attentions[0].size(1)
    seq_len = attentions[0].size(2)
    num_layers = len(attentions)
    
    flow_matrices = []
    
    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores for parallel Max-Flow computation...")
    
    for b in range(batch_size):
        print(f"Computing Per-Head Attention Flow for sequence {b+1}/{batch_size}...")
        head_flow_matrices = []
        
        for h in range(num_heads):
            print(f"  -> Processing Head {h+1}/{num_heads}...")
            
            # 1. Compute normalized attention matrices for this specific head across all layers
            norm_attns = []
            for l in range(num_layers):
                # We do NOT average. We pick the specific head h.
                layer_attn = attentions[l][b][h]
                identity = torch.eye(seq_len).to(layer_attn.device)
                attn_with_res = layer_attn + identity
                
                row_sums = attn_with_res.sum(dim=-1, keepdim=True)
                row_sums = torch.clamp(row_sums, min=1e-9)
                norm_attn = attn_with_res / row_sums
                norm_attns.append(norm_attn.detach().cpu().numpy())
                
            # 2. Build the Directed Acyclic Graph (DAG) for Max-Flow
            G = nx.DiGraph()
            for l in range(num_layers):
                A = norm_attns[l]
                for i in range(seq_len):
                    for j in range(seq_len):
                        if A[i, j] > 1e-5: 
                            u = f"L{l}_{j}"
                            v = f"L{l+1}_{i}"
                            G.add_edge(u, v, capacity=A[i, j])
                            
            # 3. Compute Max Flow for all pairs using Multiprocessing
            flow_matrix = np.zeros((seq_len, seq_len))
            
            func = partial(_compute_single_target_flow, seq_len=seq_len, num_layers=num_layers, G=G)
            
            with Pool(processes=num_cores) as pool:
                results = list(tqdm(pool.imap(func, range(seq_len)), total=seq_len, desc=f"Calculating Max Flow (Head {h+1})"))
                
            for i, row in results:
                flow_matrix[i, :] = row
                        
            head_flow_matrices.append(flow_matrix)
            
        flow_matrices.append(np.array(head_flow_matrices))
        
    return np.array(flow_matrices)

def plot_flow_heatmaps(flow_matrix, item_ids, save_dir="attention_plots", seq_idx=0):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Convert item IDs to strings for labels
    labels = [str(int(x)) for x in item_ids]
    
    sns.heatmap(flow_matrix, cmap="viridis", ax=ax, cbar=True, vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels)
    ax.set_title("Attention Flow (Max-Flow Formulation)")
    ax.set_ylabel("Target (Query) Item ID")
    ax.set_xlabel("Source (Key) Item ID")
    
    ax.tick_params(axis='x', labelsize=6, rotation=90)
    ax.tick_params(axis='y', labelsize=6, rotation=0)
            
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"attention_flow_seq_{seq_idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Attention Flow heatmap saved to {save_path}")

def plot_per_head_flow_heatmaps(flow_matrix, item_ids, save_dir="attention_plots", seq_idx=0):
    os.makedirs(save_dir, exist_ok=True)
    num_heads = flow_matrix.shape[0]
    
    fig, axes = plt.subplots(1, num_heads, figsize=(20 * num_heads, 16), squeeze=False)
    
    labels = [str(int(x)) for x in item_ids]
    
    for head_idx in range(num_heads):
        ax = axes[0, head_idx]
        sns.heatmap(flow_matrix[head_idx], cmap="viridis", ax=ax, cbar=True, vmin=0, vmax=1,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(f"Attention Flow - Head {head_idx + 1}")
        ax.set_ylabel("Target (Query) Item ID")
        ax.set_xlabel("Source (Key) Item ID")
        
        ax.tick_params(axis='x', labelsize=6, rotation=90)
        ax.tick_params(axis='y', labelsize=6, rotation=0)
            
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"attention_flow_per_head_seq_{seq_idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Per-Head Attention Flow heatmap saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract and visualize Attention Flow")
    parser.add_argument('--config', type=str, default='config_ml1m_sasrec.py', help='Configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--save_dir', type=str, default='attention_plots', help='Directory to save the heatmaps')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of sequences to visualize from the test set')
    parser.add_argument('--per_head', action='store_true', help='Compute Attention Flow independently for each head')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    
    model = build_model(config)
    model = model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    test_dataloader = get_test_dataloader(config.dataset_name, batch_size=args.num_samples, max_length=config.sequence_length)
    batch = next(iter(test_dataloader))
    input_seqs = batch[0].to(device)
    
    with torch.no_grad():
        _, attentions = model(input_seqs)
        
    if args.per_head:
        flow_matrices = compute_per_head_attention_flow(attentions)
        for i in range(flow_matrices.shape[0]):
            plot_per_head_flow_heatmaps(flow_matrices[i], input_seqs[i], save_dir=args.save_dir, seq_idx=i)
    else:
        flow_matrices = compute_attention_flow(attentions)
        for i in range(flow_matrices.shape[0]):
            plot_flow_heatmaps(flow_matrices[i], input_seqs[i], save_dir=args.save_dir, seq_idx=i)
        
if __name__ == '__main__':
    main()
