import argparse
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from dataset_utils import get_test_dataloader
from utils import build_model, get_device, load_config

def get_logits(model, input_seqs):
    """
    Runs a forward pass and computes the final item logits.
    """
    seq_emb, _ = model(input_seqs)
    
    # We only care about predicting the *next* item, so we take the hidden state
    # at the final sequence step for the batch
    final_seq_emb = seq_emb[:, -1, :]
    
    # Project hidden states to vocabulary logits
    output_embeddings = model.get_output_embeddings()
    scores = torch.einsum('bd,nd->bn', final_seq_emb, output_embeddings.weight)
    
    # Mask out padding token and item 0 (as done in the model's get_predictions)
    scores[:, 0] = float("-inf")
    scores[:, model.num_items + 1:] = float("-inf")
    
    return scores

def get_ablation_hook(target_position):
    """
    Returns a PyTorch forward hook that zero-ablates the hidden state
    at a specific sequence position. Robustly handles both single tensors
    and tuples (like MHA or full layer outputs).
    """
    def hook(module, input, output):
        # Check if output is a tuple (e.g., from TransformerBlock or MultiHeadAttention)
        if isinstance(output, tuple):
            seq = output[0]
            # CRITICAL: We must clone to avoid PyTorch in-place operation errors
            patched_seq = seq.clone()
            
            # Zero-Ablation: Destroy all information at the target sequence position
            patched_seq[:, target_position, :] = 0.0
            
            # Reconstruct the tuple with the patched tensor
            return (patched_seq,) + output[1:]
        else:
            # Output is a single tensor (e.g., from an FFN Linear layer)
            patched_seq = output.clone()
            
            # Zero-Ablation
            patched_seq[:, target_position, :] = 0.0
            
            return patched_seq
            
    return hook

def main():
    parser = argparse.ArgumentParser(description="Causal Tracing via Activation Patching (Zero-Ablation)")
    parser.add_argument('--config', type=str, default='config_ml1m_sasrec.py', help='Configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--save_dir', type=str, default='causal_plots', help='Directory to save the heatmaps')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of sequences to visualize from the test set')
    parser.add_argument('--patch_target', type=str, choices=['layer', 'mha', 'ffn'], default='layer',
                        help='Which submodule to ablate (layer=full block, mha=attention, ffn=feed-forward)')
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
    
    # Get a single batch of size 1
    batch = next(iter(test_dataloader))
    input_seqs = batch[0].to(device)
    
    seq_len = input_seqs.size(1)
    num_layers = len(model.transformer_blocks)

    print(f"Processing sequence of length {seq_len} across {num_layers} layers.")

    # Clean Run (Baseline)
    with torch.no_grad():
        clean_logits = get_logits(model, input_seqs)
        
        # Identify the top-1 recommended item (argmax)
        top_item_idx = torch.argmax(clean_logits, dim=-1).item()
        clean_logit_value = clean_logits[0, top_item_idx].item()
        
    print(f"Baseline Clean Run -> Predicted Item: {top_item_idx} | Logit: {clean_logit_value:.4f}")

    # Systematic Patching Loop
    # Initialize a 2D matrix to store the causal logit drops
    causal_heatmap_matrix = np.zeros((num_layers, seq_len))
    
    for l in range(num_layers):
        # Dynamically select the target module based on user input
        if args.patch_target == 'layer':
            target_module = model.transformer_blocks[l]
        elif args.patch_target == 'mha':
            # Submodule name in your TransformerBlock implementation
            target_module = model.transformer_blocks[l].multihead_attention
        elif args.patch_target == 'ffn':
            # In SASRec, the FFN is built with dense1 and dense2 inside the block.
            # Ablating dense2 safely zeros out the FFN's output right before the residual connection.
            target_module = model.transformer_blocks[l].dense2
            
        for p in tqdm(range(seq_len), desc=f"Ablating Layer {l} ({args.patch_target})"):
            # Register the hook on the specific submodule
            hook_handle = target_module.register_forward_hook(get_ablation_hook(target_position=p))
            
            try:
                with torch.no_grad():
                    # Run a forward pass with the hook active
                    patched_logits = get_logits(model, input_seqs)
                    
                    # Extract the logit of the SAME top-1 item
                    patched_logit_value = patched_logits[0, top_item_idx].item()
                    
                    # Calculate the Direct Logit Drop
                    logit_drop = clean_logit_value - patched_logit_value
                    
                    # Store it
                    causal_heatmap_matrix[l, p] = logit_drop
            finally:
                # Safely remove the hook so it doesn't affect future passes
                hook_handle.remove()
                
    # Visualization
    print("Generating Causal Tracing Heatmap...")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(20, max(4, 2 * num_layers)))
    
    # Get Item IDs for the X-axis labels
    item_ids = [str(int(x)) for x in input_seqs[0].cpu().numpy()]
    layer_labels = [f"Layer {l}" for l in range(num_layers)]
    
    # Plot the heatmap
    sns.heatmap(causal_heatmap_matrix, cmap="coolwarm", center=0, ax=ax, cbar=True,
                xticklabels=item_ids, yticklabels=layer_labels)
                
    ax.set_title(f"Causal Tracing (Zero-Ablation) | Target: {args.patch_target.upper()}\nTarget Predicted Item: {top_item_idx}")
    ax.set_ylabel("Transformer Block Ablated")
    ax.set_xlabel("Ablated Sequence Position (Item ID)")
    
    # Adjust tick params for readability
    ax.tick_params(axis='x', labelsize=6, rotation=90)
    ax.tick_params(axis='y', labelsize=10, rotation=0)
    
    plt.tight_layout()
    save_path = os.path.join(args.save_dir, f"causal_tracing_heatmap_{args.patch_target}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Causal Tracing Heatmap saved to {save_path}")

if __name__ == '__main__':
    main()
