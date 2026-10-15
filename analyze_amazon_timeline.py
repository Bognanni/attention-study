import os
import json
import gzip
import shutil
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import argparse

# All 34 categories available in Amazon Reviews 2023 (benchmark 0core rating_only)
ALL_AMAZON_2023_CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
    "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
    "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food",
    "Handmade_Products", "Health_and_Household", "Health_and_Personal_Care",
    "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
    "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments",
    "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Software",
    "Sports_and_Outdoors", "Subscription_Boxes", "Tools_and_Home_Improvement",
    "Toys_and_Games", "Unknown", "Video_Games"
]

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only"

def download_category_if_needed(category, cache_dir):
    cache_path = cache_dir / f"{category}.csv.gz"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    
    url = f"{BASE_URL}/{category}.csv.gz"
    print(f"Downloading {category} from {url}...")
    temp_path = cache_dir / f"{category}.csv.gz.part"
    try:
        urllib.request.urlretrieve(url, temp_path)
        shutil.move(temp_path, cache_path)
        print(f"Saved {category} ({cache_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {category}: {e}")
    return cache_path

def stream_interactions_in_window(file_path, t_min, t_max):
    """
    Streams (user, item, timestamp) triples within [t_min, t_max] from either
    a CSV (.csv.gz) or JSONL (.jsonl or .json.gz) Amazon 2023 file.
    Only rows within [t_min, t_max] are loaded into RAM.
    """
    interactions = []
    file_str = str(file_path)
    
    # Handle CSV.GZ
    if file_str.endswith(".csv.gz"):
        with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split(",")
            try:
                user_idx = header.index("user_id")
                item_idx = header.index("parent_asin") if "parent_asin" in header else header.index("asin")
                time_idx = header.index("timestamp")
            except ValueError:
                user_idx, item_idx, time_idx = 0, 1, 3

            for line in f:
                parts = line.strip().split(",")
                if len(parts) <= max(user_idx, item_idx, time_idx):
                    continue
                try:
                    raw_time = float(parts[time_idx])
                    t = raw_time / 1000.0 if raw_time > 1e11 else raw_time
                    if t_min <= t <= t_max:
                        user = parts[user_idx].strip()
                        item = parts[item_idx].strip()
                        if user and item:
                            interactions.append((user, item, t))
                except ValueError:
                    continue

    # Handle JSONL / JSON.GZ
    elif file_str.endswith(".jsonl") or file_str.endswith(".json.gz"):
        opener = gzip.open if file_str.endswith(".gz") else open
        with opener(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    user = data.get("user_id") or data.get("reviewerID")
                    item = data.get("parent_asin") or data.get("asin") or data.get("item_id")
                    raw_time = data.get("timestamp") or data.get("unixReviewTime")
                    if user and item and raw_time is not None:
                        t = float(raw_time)
                        t = t / 1000.0 if t > 1e11 else t
                        if t_min <= t <= t_max:
                            interactions.append((user, item, t))
                except (json.JSONDecodeError, ValueError):
                    continue

    return interactions

def analyze_timeline(interactions, t_end, test_days=15, max_train_days=360, step_days=30, min_freq_warm=5):
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

    print("\n" + "="*125)
    print("EXPLORATORY TIMELINE & K-CORE ANALYSIS (PETROV & MACDONALD 2022)")
    print(f"Reference End Date (T_end):   {datetime.fromtimestamp(t_end, tz=timezone.utc)}")
    print(f"Global Split Date  (T_split): {datetime.fromtimestamp(t_split, tz=timezone.utc)} ({test_days} days test window)")
    print(f"Min Freq for Warm Item:       >= {min_freq_warm}")
    print("="*125)

    # 1. Total interactions in the test window (t_split, t_end]
    total_test_inters = sum(1 for u, i, t in interactions if t_split < t <= t_end)

    # 2. Extract test targets: for users with history before t_split, find their first interaction after t_split
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

    print("="*125 + "\n")
    print("TIPS FOR SELECTION:")
    print("  • Inspect the Warm Ratio plateau to detect when adding more days brings diminishing returns.")
    print("  • Check the 5-Core volume: ensure you have enough training sequences (e.g. >= 50,000) for stable SASRec attention.")
    print("  • To generate the dataset with your chosen duration and core, run 'build_amazon_gts_dataset.py'.\n")

def main():
    parser = argparse.ArgumentParser(description="Exploratory Timeline Analysis for Amazon 2023 (Global Timeline Split)")
    parser.add_argument("--categories", nargs="+", default=["All_Beauty"],
                        help="List of Amazon 2023 categories to include or 'all' for all 34 categories.")
    parser.add_argument("--local_file", type=str, default=None,
                        help="Optional direct path to a local CSV or JSONL file (e.g. All_Beauty.jsonl/All_Beauty.jsonl).")
    parser.add_argument("--cache_dir", type=str, default="raw_amazon_2023",
                        help="Local cache directory for downloaded raw files.")
    parser.add_argument("--test_days", type=int, default=15,
                        help="Duration of the test window in days (default: 15).")
    parser.add_argument("--max_train_days", type=int, default=360,
                        help="Maximum training window duration to test in days (default: 360).")
    parser.add_argument("--step_days", type=int, default=30,
                        help="Increment step for candidate training durations in days (default: 30).")
    parser.add_argument("--min_freq_warm", type=int, default=5,
                        help="Minimum training appearances to consider an item warm (default: 5).")
    parser.add_argument("--t_end", type=float, default=None,
                        help="Optional fixed end timestamp. Defaults to 2023-09-09 00:00:00 UTC.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    files_to_process = []
    if args.local_file and os.path.exists(args.local_file):
        files_to_process.append(Path(args.local_file))
    else:
        cats = ALL_AMAZON_2023_CATEGORIES if "all" in args.categories else args.categories
        for c in cats:
            local_cand = Path(f"{c}.jsonl/{c}.jsonl")
            if local_cand.exists():
                files_to_process.append(local_cand)
            else:
                files_to_process.append(download_category_if_needed(c, cache_dir))

    DEFAULT_AMAZON_2023_END = 1694227200.0
    t_end = args.t_end if args.t_end is not None else DEFAULT_AMAZON_2023_END
    print(f"Using reference end timestamp: {t_end} ({datetime.fromtimestamp(t_end, tz=timezone.utc)})")

    # Load candidate interactions up to (max_train_days + test_days)
    candidate_window_start = t_end - ((args.max_train_days + args.test_days) * 86400)
    print(f"Loading candidate interactions from {len(files_to_process)} source(s) in window [{candidate_window_start}, {t_end}]...")

    all_interactions = []
    for f in files_to_process:
        print(f"Reading from {f.name}...")
        sub = stream_interactions_in_window(f, candidate_window_start, t_end)
        all_interactions.extend(sub)
        print(f"  Extracted {len(sub):,} interactions within window.")

    if not all_interactions:
        print("ERROR: No interactions found within the specified time window!")
        return

    # Chronological sort and deduplication of identical triples
    print(f"Sorting and deduplicating {len(all_interactions):,} interactions...")
    seen = set()
    dedup_interactions = []
    for u, i, t in all_interactions:
        key = (u, i, t)
        if key not in seen:
            seen.add(key)
            dedup_interactions.append((u, i, t))
    
    dedup_interactions.sort(key=lambda x: x[2])
    print(f"Retained {len(dedup_interactions):,} unique interactions.")

    analyze_timeline(
        dedup_interactions,
        t_end=t_end,
        test_days=args.test_days,
        max_train_days=args.max_train_days,
        step_days=args.step_days,
        min_freq_warm=args.min_freq_warm
    )

if __name__ == "__main__":
    main()
