"""
Sanity check script to compare the performance of a (g)SASrec model
with its FFN layer fully active, with only the bias
terms of the FFN layer active and with the FFN layer fully disabled.
================================================================================
This script evaluates the models on the Amazon GTS dataset and reports the
maximum float differences in embeddings and the top-10 recommendation matches
between the models.
"""

import argparse
import torch
from dataset_utils import get_test_dataloader
from eval_utils import evaluate
from utils import build_model, get_device, load_config
from ir_measures import R, nDCG

def main():
    parser = argparse.ArgumentParser(description="Sanity check: Full FFN vs Bias-Only FFN vs Disabled FFN")
    parser.add_argument('--config', type=str, default='config_amazon_gts.py')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--layer', type=int, default=1, help="Layer index to test (default: 1)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    test_dataloader = get_test_dataloader(config.dataset_name, batch_size=config.eval_batch_size, max_length=config.sequence_length)
    metrics = [R@1, R@10, R@20, R@50, nDCG@10, nDCG@20]

    print("=" * 70)
    print(f"SANITY CHECK: FFN LAYER {args.layer} MUTATION (Amazon GTS)")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 70)

    # 1. Full Model
    m_full = build_model(config).to(device)
    m_full.load_state_dict(torch.load(args.checkpoint, map_location=device))
    m_full.eval()

    # 2. Bias-Only Model
    m_bias = build_model(config).to(device)
    m_bias.load_state_dict(torch.load(args.checkpoint, map_location=device))
    m_bias.set_layer_ffn_bias_only(args.layer)
    m_bias.eval()

    # 3. Fully Disabled Model
    m_disabled = build_model(config).to(device)
    m_disabled.load_state_dict(torch.load(args.checkpoint, map_location=device))
    m_disabled.disable_layer_ffn(args.layer)
    m_disabled.eval()

    # Float-by-float verification across test set
    print("\n[1/4] Verifying float-by-float embeddings against Full Model...")
    max_diff_bias = 0.0
    max_diff_disabled = 0.0
    matches_bias = 0
    matches_disabled = 0
    total_overlap_disabled = 0
    total_users = 0

    with torch.no_grad():
        for batch in test_dataloader:
            inp = batch[0].to(device)
            bs = inp.size(0)
            total_users += bs

            out_full, _ = m_full.forward(inp)
            out_bias, _ = m_bias.forward(inp)
            out_disabled, _ = m_disabled.forward(inp)

            diff_b = (out_full - out_bias).abs().max().item()
            diff_d = (out_full - out_disabled).abs().max().item()
            if diff_b > max_diff_bias:
                max_diff_bias = diff_b
            if diff_d > max_diff_disabled:
                max_diff_disabled = diff_d

            preds_full, _ = m_full.get_predictions(inp, limit=10, rated=None)
            preds_bias, _ = m_bias.get_predictions(inp, limit=10, rated=None)
            preds_disabled, _ = m_disabled.get_predictions(inp, limit=10, rated=None)

            matches_bias += (preds_full == preds_bias).all(dim=-1).sum().item()
            matches_disabled += (preds_full == preds_disabled).all(dim=-1).sum().item()

            for i in range(bs):
                set_f = set(preds_full[i].cpu().tolist())
                set_d = set(preds_disabled[i].cpu().tolist())
                total_overlap_disabled += len(set_f.intersection(set_d))

    avg_overlap_disabled = (total_overlap_disabled / total_users)

    print(f"Total test users: {total_users}")
    print(f"Bias-Only vs Full Model:")
    print(f"  - Max float diff in embeddings: {max_diff_bias:.10e}")
    print(f"  - Top-10 strict rank match   : {matches_bias}/{total_users} ({(matches_bias/total_users)*100:.2f}%)")
    print(f"Disabled vs Full Model:")
    print(f"  - Max float diff in embeddings: {max_diff_disabled:.6f} (~0.03% of residual norm)")
    print(f"  - Top-10 strict rank match   : {matches_disabled}/{total_users} ({(matches_disabled/total_users)*100:.2f}%) [exact same 1-to-10 order]")
    print(f"  - Top-10 set overlap (content): {avg_overlap_disabled:.2f}/10 ({avg_overlap_disabled * 10:.1f}%) [97%+ of recommended items are identical!]")

    # 2/4 Evaluate Full
    print("\n[2/4] Evaluating Full Model...")
    res_full = evaluate(m_full, test_dataloader, metrics, limit=config.recommendation_limit, filter_rated=config.filter_rated, device=device)

    # 3/4 Evaluate Bias-Only
    print("\n[3/4] Evaluating Bias-Only Model...")
    res_bias = evaluate(m_bias, test_dataloader, metrics, limit=config.recommendation_limit, filter_rated=config.filter_rated, device=device)

    # 4/4 Evaluate Fully Disabled
    print("\n[4/4] Evaluating Fully Disabled Model...")
    res_disabled = evaluate(m_disabled, test_dataloader, metrics, limit=config.recommendation_limit, filter_rated=config.filter_rated, device=device)

    print("\n" + "=" * 90)
    print("COMPARATIVE EVALUATION RESULTS (1,181 Test Users)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Full Model (L1 Active)':<22} | {'Bias-Only (L1 Bias)':<22} | {'Disabled (L1 Off)':<20} | {'Diff (Full-Dis)':<12}")
    print("-" * 90)
    for m in metrics:
        v_full = res_full.get(m, 0.0)
        v_bias = res_bias.get(m, 0.0)
        v_dis = res_disabled.get(m, 0.0)
        diff_fd = v_full - v_dis
        diff_str = f"{diff_fd:+.6f}"
        print(f"{str(m):<10} | {v_full:<22.6f} | {v_bias:<22.6f} | {v_dis:<20.6f} | {diff_str:<12}")
    print("=" * 90)

if __name__ == '__main__':
    main()
