import torch
import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr

class FaithfulnessEvaluator:
    """
    Computes statistical faithfulness metrics comparing a heuristic ranking
    (like Attention Rollout) against a True Causal ranking.
    """
    @staticmethod
    def compute_spearman(heuristic_scores, causal_scores, valid_idx):
        """
        Computes the Spearman rank correlation coefficient between the heuristic scores.
        """
        h_valid = heuristic_scores[valid_idx].cpu().numpy()
        c_valid = causal_scores[valid_idx].cpu().numpy()
        
        if len(h_valid) < 2: return 0.0
        
        spearman, _ = spearmanr(h_valid, c_valid)
        return 0.0 if np.isnan(spearman) else float(spearman)
        
    @staticmethod
    def compute_jaccard(heuristic_scores, causal_scores, valid_idx, k=3):
        """
        Computes the Jaccard similarity between the top-k items of the heuristic and causal rankings.
        """
        h_valid = heuristic_scores[valid_idx].cpu().numpy()
        c_valid = causal_scores[valid_idx].cpu().numpy()
        
        actual_k = min(k, len(h_valid))
        if actual_k == 0: return 0.0

        # Order the indices of the valid heuristic and causal scores to get the top-k items
        h_topk = set(np.argsort(h_valid)[-actual_k:])
        c_topk = set(np.argsort(c_valid)[-actual_k:])
        
        intersection = len(h_topk.intersection(c_topk))
        union = len(h_topk.union(c_topk))
        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def compute_comprehensiveness_and_sufficiency(model, input_seq, target_item, pad_token_id, valid_idx, ranking_scores, k=3):
        """
        Comprehensiveness: Drop in target probability when Top-K are masked.
        Sufficiency: Retention of target probability when ONLY Top-K are kept.
        """
        scores = ranking_scores[valid_idx].cpu().numpy()
        actual_k = min(k, len(scores))
        if actual_k == 0: return 0.0, 0.0
        
        # Get the original sequence indices of the Top-K items
        topk_relative_idx = np.argsort(scores)[-actual_k:]
        topk_idx = valid_idx[topk_relative_idx]
        
        seq_2d = input_seq.unsqueeze(0)
        with torch.no_grad():
            base_logits = FaithfulnessEvaluator.get_logits(model, seq_2d)
            base_logit = base_logits[0, target_item].item()
            
        # Hybrid masking rule: padding for history (clean ablation), random token for T-0 (prevent collapse)
        def get_masked_value(idx):
            if idx == valid_idx[-1].item():
                return torch.randint(1, model.num_items + 1, (1,), device=input_seq.device).item()
            return pad_token_id
            
        # Fixed Comprehensiveness: Mask Top-K
        comp_seq = input_seq.clone()
        for idx in topk_idx:
            comp_seq[idx] = get_masked_value(idx)
            
        with torch.no_grad():
            comp_logits = FaithfulnessEvaluator.get_logits(model, comp_seq.unsqueeze(0))
            comp_logit = comp_logits[0, target_item].item()

        # higher is better
        comprehensiveness = base_logit - comp_logit
        
        # Sufficiency: Keep Top-K
        suff_seq = input_seq.clone()
        for idx in valid_idx:
            if idx not in topk_idx:
                suff_seq[idx] = get_masked_value(idx)
                
        with torch.no_grad():
            suff_logits = FaithfulnessEvaluator.get_logits(model, suff_seq.unsqueeze(0))
            suff_logit = suff_logits[0, target_item].item()

        # lower is better
        sufficiency = base_logit - suff_logit
        
        return comprehensiveness, sufficiency

    @staticmethod
    def get_logits(model, input_seqs):
        """
        Computes the logits for the given input sequences and for each item in the catalog using the model.
        """
        seq_emb, _ = model(input_seqs)
        final_seq_emb = seq_emb[:, -1, :]
        output_embeddings = model.get_output_embeddings()
        scores = torch.einsum('bd,nd->bn', final_seq_emb, output_embeddings.weight)
        scores[:, 0] = float("-inf")
        scores[:, model.num_items + 1:] = float("-inf")
        return scores

    @staticmethod
    def compute_loo_ablation(model, input_seq, target_item, pad_token_id):
        """
        Leave-One-Out (LOO) Input Ablation.
        Returns the causal drops (using logits), a boolean mask of highly causal items
        (using Softmax Probability drop > 50%), and the raw prob_drop_ratios.
        """
        valid_idx = (input_seq != pad_token_id).nonzero(as_tuple=True)[0]
        causal_drops = torch.zeros(len(input_seq), device=input_seq.device)
        is_highly_causal = torch.zeros(len(input_seq), dtype=torch.bool, device=input_seq.device)
        prob_drop_ratios = torch.zeros(len(input_seq), dtype=torch.float32, device=input_seq.device)
        
        seq_2d = input_seq.unsqueeze(0)
        with torch.no_grad():
            base_logits = FaithfulnessEvaluator.get_logits(model, seq_2d)
            base_logit = base_logits[0, target_item].item()
            
            base_probs = torch.softmax(base_logits, dim=-1)
            base_prob = base_probs[0, target_item].item()
            
        for idx in valid_idx:
            ablated_seq = input_seq.clone()
            if idx == valid_idx[-1].item():
                masked_val = torch.randint(1, model.num_items + 1, (1,), device=input_seq.device).item()
            else:
                masked_val = pad_token_id
            ablated_seq[idx] = masked_val
            
            with torch.no_grad():
                ablated_logits = FaithfulnessEvaluator.get_logits(model, ablated_seq.unsqueeze(0))
                ablated_logit = ablated_logits[0, target_item].item()
                
                ablated_probs = torch.softmax(ablated_logits, dim=-1)
                ablated_prob = ablated_probs[0, target_item].item()
                
            # Logit drop for linear Causal Ranking
            causal_drops[idx] = base_logit - ablated_logit
            
            # Probability drop for identifying "Vital/Highly Causal" items (for the second part)
            prob_drop_ratio = (base_prob - ablated_prob) / base_prob if base_prob > 0 else 0
            prob_drop_ratios[idx] = prob_drop_ratio
            
            if prob_drop_ratio > 0.50:
                is_highly_causal[idx] = True
                
        return causal_drops, is_highly_causal, valid_idx, prob_drop_ratios

def calculate_popularity_baseline(dataset_name, head_percentage=0.20):
    """
    Computes the popularity baseline for the dataset.
    Returns the set of head items and the ratio of interactions they account for.
    """
    train_file = f"datasets/{dataset_name}/train/input.txt"
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
                    
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        num_unique_items = len(sorted_items)
        head_cutoff = int(num_unique_items * head_percentage)
        # ids of the top 20% most popular items
        head_items_set = set([item[0] for item in sorted_items[:head_cutoff]])

        # interaction of head items / total interactions
        head_interactions = sum([item[1] for item in sorted_items[:head_cutoff]])
        dataset_head_ratio = head_interactions / total_train_interactions if total_train_interactions > 0 else 0
        return head_items_set, dataset_head_ratio
    else:
        print(f"\nWarning: Train file not found at {train_file}. Skipping popularity baseline.")
        return set(), 0.0

def get_item_popularity_bins(dataset_name):
    """
    Groups all items into 10 deciles based on their popularity (frequency).
    Decile 1 = Top 10% most popular items.
    Returns:
    - item_to_bin: mapping item_id -> decile_string (e.g. 'Top 10-20%').
    - item_to_percentile: mapping item_id -> exact percentile float (0.0 to 100.0)
    """
    train_file = f"datasets/{dataset_name}/train/input.txt"
    item_counts = {}
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            for line in f:
                for item in line.strip().split():
                    item_id = int(item)
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1
                    
        sorted_items = sorted(item_counts.keys(), key=lambda x: item_counts[x], reverse=True)
        num_items = len(sorted_items)
        
        item_to_bin = {}
        item_to_percentile = {}
        for rank, item_id in enumerate(sorted_items):
            percentile = (rank / num_items) * 100
            # Find which decile it belongs to
            decile_idx = int(percentile // 10)
            if decile_idx == 10: decile_idx = 9 # edge case for exactly 100%
            
            bin_name = f"Top {decile_idx*10}-{(decile_idx+1)*10}%"
            item_to_bin[item_id] = bin_name
            item_to_percentile[item_id] = percentile
            
        return item_to_bin, item_to_percentile
    return {}, {}


def compute_user_overlap(categories=["Video_Games", "Toys_and_Games", "Electronics"]):
    """
    Computes the overlap of users between different Amazon categories (2023).
    """

    user_sets = {}

    for cat in categories:
        url = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/{cat}.csv.gz"
        df = pd.read_csv(url, usecols=["user_id"])
        user_sets[cat] = set(df["user_id"].unique())
        print(f"{cat}: {len(user_sets[cat]):,} unique users")

    # Compute the overlap between each pair of categories
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            c1, c2 = categories[i], categories[j]
            u1, u2 = user_sets[c1], user_sets[c2]
            inter = len(u1 & u2)
            jaccard = inter / len(u1 | u2) * 100
            containment = inter / min(len(u1), len(u2)) * 100
            print(
                f"\nOverlap {c1} <-> {c2}: {inter:,} shared users | Jaccard:"
                f" {jaccard:.2f}% | On smaller set: {containment:.2f}%"
            )
            
    # Compute the global overlap across ALL categories
    if len(categories) > 2:
        all_intersection = set.intersection(*user_sets.values())
        all_union = set.union(*user_sets.values())
        min_set_size = min(len(u) for u in user_sets.values())
        
        global_inter = len(all_intersection)
        global_jaccard = (global_inter / len(all_union)) * 100
        global_containment = (global_inter / min_set_size) * 100
        
        cats_str = " <-> ".join(categories)
        print(f"\n[GLOBAL OVERLAP] {cats_str}")
        print(f"  Users present in ALL datasets: {global_inter:,} | Global Jaccard: {global_jaccard:.2f}% | On smallest set: {global_containment:.2f}%")

# compute_user_overlap()