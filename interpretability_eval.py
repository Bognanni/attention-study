import torch
import numpy as np
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
        Computes the Jaccard similarity between the top-k items of the heuristic and causal rankings."""
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
            
        # Comprehensiveness: Mask Top-K
        comp_seq = input_seq.clone()
        for idx in topk_idx:
            comp_seq[idx] = get_masked_value(idx)
            
        with torch.no_grad():
            comp_logits = FaithfulnessEvaluator.get_logits(model, comp_seq.unsqueeze(0))
            comp_logit = comp_logits[0, target_item].item()
            
        comprehensiveness = base_logit - comp_logit
        
        # Sufficiency: Keep Top-K
        suff_seq = input_seq.clone()
        for idx in valid_idx:
            if idx not in topk_idx:
                suff_seq[idx] = get_masked_value(idx)
                
        with torch.no_grad():
            suff_logits = FaithfulnessEvaluator.get_logits(model, suff_seq.unsqueeze(0))
            suff_logit = suff_logits[0, target_item].item()
            
        sufficiency = base_logit - suff_logit
        
        return comprehensiveness, sufficiency

    @staticmethod
    def get_logits(model, input_seqs):
        """
        Computes the logits for the given input sequences and for each item in the catalog using the model."""
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
        Returns the causal drops (using logits) and a boolean mask of highly causal items
        (using Softmax Probability drop > 50%).
        """
        valid_idx = (input_seq != pad_token_id).nonzero(as_tuple=True)[0]
        causal_drops = torch.zeros(len(input_seq), device=input_seq.device)
        is_highly_causal = torch.zeros(len(input_seq), dtype=torch.bool, device=input_seq.device)
        
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
            if prob_drop_ratio > 0.50:
                is_highly_causal[idx] = True
                
        return causal_drops, is_highly_causal, valid_idx
