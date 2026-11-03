import os
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

from gts_core import (
    stream_canonical_interactions,
    ALL_AMAZON_2023_CATEGORIES,
    download_amazon_category
)


def analyze_timeline_matrix(interactions, t_end, test_days=15, max_train_days=360, step_days=30, min_freq_warm=5):
    """
    Performs exploratory calibration by testing training durations incremented by step_days.
    Computes:
      - Train Inters and % Train vs % Test (sum = 100.00%)
      - Warm Ratio on Test Targets
      - Sequence volume across multiple k-cores (1, 2, 3, 5, 10)
    Does NOT export or modify any dataset files.
    """
    day_sec = 86400
    t_split = t_end - (test_days * day_sec)

    print("\n" + "=" * 125)
    print("UNIVERSAL EXPLORATORY TIMELINE & K-CORE ANALYSIS (PETROV & MACDONALD 2022)")
    print(f"Reference End Date (T_end):   {datetime.fromtimestamp(t_end, tz=timezone.utc)}")
    print(f"Global Split Date  (T_split): {datetime.fromtimestamp(t_split, tz=timezone.utc)} ({test_days} days test window)")
    print(f"Min Freq for Warm Item:       >= {min_freq_warm}")
    print("=" * 125)

    # Total interactions in the test window (t_split, t_end]
    total_test_inters = sum(1 for u, i, t in interactions if t_split < t <= t_end)

    # Extract test targets: for users with history before t_split, find their first interaction after t_split
    users_with_past = {u for u, i, t in interactions if t < t_split}
    test_user_targets = {}
    for u, i, t in interactions:
        if u in users_with_past and t_split < t <= t_end:
            if u not in test_user_targets or t < test_user_targets[u][1]:
                test_user_targets[u] = (i, t)
                
    test_targets = [item for item, _ in test_user_targets.values()]
    total_test_targets = len(test_targets)

    if total_test_targets == 0:
        print("WARNING: No users with past history found in the test window.")
        return

    # Candidate durations
    candidate_train_days = list(range(step_days, max_train_days + 1, step_days))

    header = (
        f"{'Train Days':<11} | {'Train Inters':<13} | {'% Train':<8} | {'% Test':<7} | "
        f"{'Warm Ratio':<11} | {'1-Core':<10} | {'2-Core':<9} | {'3-Core':<9} | {'5-Core':<9} | {'10-Core':<9} | {'Test Targets'}"
    )
    print(header)
    print("-" * 125)

    for days in candidate_train_days:
        t_start_candidate = t_split - (days * day_sec)
        
        # Count item frequency strictly in [t_start_candidate, t_split)
        train_counts = Counter()
        for u, i, t in interactions:
            if t_start_candidate <= t < t_split:
                train_counts[i] += 1
                
        train_inters = sum(train_counts.values())
        tot_inters = train_inters + total_test_inters
        pct_train = (train_inters / tot_inters) * 100.0 if tot_inters > 0 else 0.0
        pct_test = (total_test_inters / tot_inters) * 100.0 if tot_inters > 0 else 0.0

        # Warm items in candidate training window
        warm_items = {item for item, c in train_counts.items() if c >= min_freq_warm}
        warm_count = sum(1 for item in test_targets if item in warm_items)
        warm_ratio = (warm_count / total_test_targets) * 100.0

        # Count warm items per user strictly within [t_start_candidate, t_split)
        user_warm_counts = Counter()
        for u, i, t in interactions:
            if t_start_candidate <= t < t_split and i in warm_items:
                user_warm_counts[u] += 1

        counts = list(user_warm_counts.values())
        core_1 = sum(1 for c in counts if c >= 1)
        core_2 = sum(1 for c in counts if c >= 2)
        core_3 = sum(1 for c in counts if c >= 3)
        core_5 = sum(1 for c in counts if c >= 5)
        core_10 = sum(1 for c in counts if c >= 10)

        row = (
            f"{str(days) + ' days':<11} | {train_inters:>13,} | {pct_train:>7.2f}% | {pct_test:>6.2f}% | "
            f"{warm_ratio:>10.2f}% | {core_1:>10,} | {core_2:>9,} | {core_3:>9,} | {core_5:>9,} | {core_10:>9,} | {total_test_targets:,}"
        )
        print(row)

    print("=" * 125 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Universal Exploratory Timeline Analysis (Global Timeline Split)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset file or directory.")
    parser.add_argument("--format", type=str, default=None, choices=["csv", "tsv", "dat", "jsonl", "parquet"],
                        help="Explicit file format. If omitted, inferred automatically.")
    parser.add_argument("--delimiter", type=str, default=None,
                        help="Delimiter for text/dat files (e.g. ',' or '::' or '\\t').")
    parser.add_argument("--user_col", type=str, default=None,
                        help="Name or 0-based index of user ID column.")
    parser.add_argument("--item_col", type=str, default=None,
                        help="Name or 0-based index of item ID column.")
    parser.add_argument("--time_col", type=str, default=None,
                        help="Name or 0-based index of timestamp column.")
    # Amazon backward compatibility options
    parser.add_argument("--categories", nargs="+", default=None,
                        help="List of Amazon 2023 categories (e.g. All_Beauty, Video_Games) or 'all'.")
    parser.add_argument("--cache_dir", type=str, default="raw_amazon_2023",
                        help="Cache directory for Amazon downloads.")
    # Timeline parameters
    parser.add_argument("--test_days", type=int, default=15,
                        help="Duration of the test window in days (default: 15).")
    parser.add_argument("--max_train_days", type=int, default=360,
                        help="Maximum training window duration to test in days (default: 360).")
    parser.add_argument("--step_days", type=int, default=30,
                        help="Increment step for candidate training durations in days (default: 30).")
    parser.add_argument("--min_freq_warm", type=int, default=5,
                        help="Minimum training appearances to consider an item warm (default: 5).")
    parser.add_argument("--t_end", type=float, default=None,
                        help="Optional reference end timestamp. Defaults to max timestamp in dataset.")
    args = parser.parse_args()

    files_to_process = []

    # Resolve files to process
    if args.data_path:
        p = Path(args.data_path)
        if p.is_dir():
            files_to_process.extend(sorted(p.glob("*.*")))
        elif p.exists():
            files_to_process.append(p)
        else:
            raise FileNotFoundError(f"Input path not found: {args.data_path}")
    elif args.categories:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cats = ALL_AMAZON_2023_CATEGORIES if "all" in args.categories else args.categories
        for c in cats:
            local_cand = Path(f"{c}.jsonl/{c}.jsonl")
            if local_cand.exists():
                files_to_process.append(local_cand)
            else:
                files_to_process.append(download_amazon_category(c, cache_dir))
    else:
        parser.print_help()
        print("\n[ERROR] No dataset source specified! Please provide either --data_path <path> or --categories <cat1> ...", flush=True)
        return

    print(f"\n[Step 1] Loading and scanning timestamps across {len(files_to_process)} source file(s)...", flush=True)
    all_interactions = []
    min_found_t = float("inf")
    max_found_t = float("-inf")

    for idx, f in enumerate(files_to_process, 1):
        print(f"  [{idx}/{len(files_to_process)}] Reading from {f.name}...", flush=True)
        sub = stream_canonical_interactions(
            file_path=f,
            file_format=args.format,
            delimiter=args.delimiter,
            user_col=args.user_col,
            item_col=args.item_col,
            time_col=args.time_col
        )
        if sub:
            t_vals = [t for _, _, t in sub]
            min_sub, max_sub = min(t_vals), max(t_vals)
            min_found_t = min(min_found_t, min_sub)
            max_found_t = max(max_found_t, max_sub)
            all_interactions.extend(sub)
        print(f"      Extracted {len(sub):,} interactions.", flush=True)

    if not all_interactions:
        print("ERROR: No valid interactions found in input files.")
        return

    print(f"\nTotal raw interactions extracted: {len(all_interactions):,}")
    print(f"Timestamp range observed: {min_found_t} ({datetime.fromtimestamp(min_found_t, tz=timezone.utc)}) to {max_found_t} ({datetime.fromtimestamp(max_found_t, tz=timezone.utc)})")

    t_end = args.t_end if args.t_end is not None else max_found_t
    print(f"Using reference end timestamp T_end: {t_end} ({datetime.fromtimestamp(t_end, tz=timezone.utc)})")

    # Chronological sort and deduplication
    print("\n[Step 2] Deduplicating interactions...", flush=True)
    seen = set()
    dedup = []
    for u, i, t in all_interactions:
        key = (u, i, t)
        if key not in seen:
            seen.add(key)
            dedup.append((u, i, t))

    dedup.sort(key=lambda x: x[2])
    print(f"Retained {len(dedup):,} unique interactions across {len({u for u, _, _ in dedup}):,} users.")

    # Analyze matrix
    analyze_timeline_matrix(
        dedup,
        t_end=t_end,
        test_days=args.test_days,
        max_train_days=args.max_train_days,
        step_days=args.step_days,
        min_freq_warm=args.min_freq_warm
    )


if __name__ == "__main__":
    main()
