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
    """
    Downloads a category csv.gz file if not already in cache_dir.
    Returns: the path to the cached file.
    """
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
    Only rows in [t_min, t_max] are loaded into RAM.
    Returns: list of (user, item, timestamp) tuples.
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
                # Fallback for alternative column naming
                user_idx, item_idx, time_idx = 0, 1, 3

            for line in f:
                parts = line.strip().split(",")
                if len(parts) <= max(user_idx, item_idx, time_idx):
                    continue
                try:
                    raw_time = float(parts[time_idx])
                    # If milliseconds, convert to seconds
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

def calibrate_timeline(interactions, t_end, test_days=7, target_warm_ratio=0.80, min_freq_warm=5):
    """
    Calibrates T_start and T_split to achieve ~80% warm targets on the test set.
    According to 'Time to Split' (Petrov & Macdonald, RecSys '22):
    - Single global split point: T_split = T_end - (test_days * 86400)
    - Train period: [T_start, T_split)
    - Test period:  (T_split, T_end]   -> Target is the first item post-T_split
    """
    day_sec = 86400
    t_split = t_end - (test_days * day_sec)
    
    print("\n" + "="*90)
    print("CALIBRATING SINGLE-CUT GLOBAL TIMELINE SPLIT (PETROV & MACDONALD 2022)")
    print(f"Reference End Date (T_end):   {datetime.fromtimestamp(t_end, tz=timezone.utc)}")
    print(f"Global Split Date  (T_split): {datetime.fromtimestamp(t_split, tz=timezone.utc)} ({test_days} days test window)")
    print(f"Target Warm Ratio:            {target_warm_ratio * 100:.1f}%")
    print(f"Min Freq for Warm Item:       >= {min_freq_warm}")
    print("="*90)

    # Identify users who have prior history before t_split
    users_with_past = {u for u, i, t in interactions if t < t_split}
    test_user_targets = {}
    for u, i, t in interactions:
        if u in users_with_past and t_split < t <= t_end:
            if u not in test_user_targets or t < test_user_targets[u][1]:
                test_user_targets[u] = (i, t)
                
    test_targets = [item for item, _ in test_user_targets.values()]
    total_test_targets = len(test_targets)
    
    if total_test_targets == 0:
        print("WARNING: No users with past history found in the test window for this subset.")
        return t_split - (30 * day_sec), t_split, t_end, 0.0

    # Try different train durations (e.g., 20, 30, 45, 60, 90, 120 days)
    candidate_train_days = [20, 30, 45, 60, 90, 120]
    best_train_days = candidate_train_days[0]
    best_ratio = 0.0

    print(f"{'Train Days':<12} | {'Train Inters':<14} | {'% Train':<9} | {'% Test':<8} | {'Warm Ratio':<12} | {'Warm Targets':<14} | {'Test Targets'}")
    print("-" * 90)
    
    for days in candidate_train_days:
        t_start_candidate = t_split - (days * day_sec)
        train_counts = Counter()
        for u, i, t in interactions:
            if t_start_candidate <= t < t_split:
                train_counts[i] += 1
                
        train_inters = sum(train_counts.values())
        tot_inters = train_inters + total_test_targets
        pct_train = (train_inters / tot_inters) * 100.0 if tot_inters > 0 else 0.0
        pct_test = (total_test_targets / tot_inters) * 100.0 if tot_inters > 0 else 0.0

        warm_items = {item for item, c in train_counts.items() if c >= min_freq_warm}
        warm_count = sum(1 for item in test_targets if item in warm_items)
        ratio = warm_count / total_test_targets
        
        print(f"{str(days) + ' days':<12} | {train_inters:>14,} | {pct_train:>8.2f}% | {pct_test:>7.2f}% | {ratio*100:>10.2f}% | {warm_count:>14,} | {total_test_targets:,}")
        
        if ratio >= target_warm_ratio:
            best_train_days = days
            best_ratio = ratio
            break
        best_train_days = days
        best_ratio = ratio

    t_start = t_split - (best_train_days * day_sec)
    print("="*90)
    print(f"SELECTED TIMELINE:")
    print(f"  T_start : {datetime.fromtimestamp(t_start, tz=timezone.utc)} (Train starts)")
    print(f"  T_split : {datetime.fromtimestamp(t_split, tz=timezone.utc)} (Train ends, Test starts)")
    print(f"  T_end   : {datetime.fromtimestamp(t_end, tz=timezone.utc)} (Test ends)")
    print(f"  Final Warm Ratio on Test Targets: {best_ratio * 100:.2f}%")
    print("="*70 + "\n")
    
    return t_start, t_split, t_end, best_ratio

def build_sequences_and_export(interactions, t_start, t_split, t_end, output_dir, 
                               val_users_count=1024, min_freq_warm=5, min_user_core=2, max_seq_len=50):
    """
    Constructs train, val, test sequences according to Single-Cut GTS:
    - Train period: [T_start, T_split)
    - Test: history < T_split, target = first item in (T_split, T_end]
    - Validation: sampled from train users, input = seq[:-1], target = seq[-1] (last item before T_split)
    """
    import numpy as np
    output_dir = Path(output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)

    # 1. Group interactions by user
    user_timeline = defaultdict(list)
    for u, i, t in interactions:
        user_timeline[u].append((t, i))

    # Sort each user chronologically
    for u in user_timeline:
        user_timeline[u].sort(key=lambda x: x[0])

    # 2. Extract train interactions up to T_split to build catalog vocabulary
    train_item_counts = Counter()
    for u, history in user_timeline.items():
        for t, i in history:
            if t_start <= t < t_split:
                train_item_counts[i] += 1

    # Keep ONLY warm items in catalog vocabulary (frequency >= min_freq_warm in train)
    warm_catalog_items = {item for item, c in train_item_counts.items() if c >= min_freq_warm}
    
    print("\n" + "="*70)
    print(f"CATALOG FILTERING (WARM THRESHOLD: >= {min_freq_warm} APPEARANCES IN TRAIN)")
    print(f"  • Raw items seen in train window: {len(train_item_counts):,}")
    print(f"  • Filtered Warm Catalog items:    {len(warm_catalog_items):,} ({(len(warm_catalog_items)/max(1, len(train_item_counts)))*100:.2f}%)")
    print("="*70)

    # Assign 1-indexed contiguous integer IDs to warm items only
    item_map = {item: idx + 1 for idx, item in enumerate(sorted(warm_catalog_items))}

    # Sequence length breakdown before filtering
    candidate_train_lengths = []
    for u, history in user_timeline.items():
        active = any(t_start <= t < t_split for t, i in history)
        if active:
            past = [item_map[i] for t, i in history if t < t_split and i in item_map]
            if len(past) > 0:
                candidate_train_lengths.append(len(past))

    print("\n" + "="*70)
    print("USER SEQUENCE LENGTH BREAKDOWN (K-CORE POTENTIAL)")
    for core in [1, 2, 3, 5, 10]:
        c_count = sum(1 for l in candidate_train_lengths if l >= core)
        pct = (c_count / max(1, len(candidate_train_lengths))) * 100
        print(f"  • {core}-core (sequence length >= {core}): {c_count:>8,} sequences ({pct:.1f}%)")
    print(f"Selected sequence length threshold (min_user_core): >= {min_user_core}")
    print("="*70 + "\n")

    # 3. Build Test Set:
    # History before T_split as input; TRUE first item in (T_split, t_end] as target
    test_inputs, test_outputs = [], []
    for u, history in user_timeline.items():
        past_items = [item_map[i] for t, i in history if t < t_split and i in item_map]
        future_events = [(t, i) for t, i in history if t_split < t <= t_end]
        
        if len(past_items) >= min_user_core and len(future_events) > 0:
            true_first_future_item = future_events[0][1] # The true first item after split
            if true_first_future_item in item_map:
                test_inputs.append(past_items[-max_seq_len:])
                test_outputs.append(item_map[true_first_future_item])

    # 4. Build Train and Validation Sets:
    # Extract candidate training users
    all_train_candidates = {}
    for u, history in user_timeline.items():
        past = [item_map[i] for t, i in history if t < t_split and i in item_map]
        active_in_train = any(t_start <= t < t_split for t, i in history)
        if len(past) >= min_user_core and active_in_train:
            all_train_candidates[u] = past

    # Select validation users from training users (requiring >= 2 items so seq[:-1] is valid)
    rng = np.random.RandomState(42)
    eligible_val_users = [u for u, seq in all_train_candidates.items() if len(seq) >= 2]
    num_val = min(val_users_count, len(eligible_val_users))
    val_user_set = set(rng.choice(eligible_val_users, num_val, replace=False)) if num_val > 0 else set()

    train_sequences = []
    val_inputs, val_outputs = [], []

    for u, seq in all_train_candidates.items():
        if u in val_user_set:
            # Mask the last item before T_split for validation
            val_inputs.append(seq[:-1][-max_seq_len:])
            val_outputs.append(seq[-1])
            # Train sequence gets the prefix without the validation target
            if len(seq[:-1]) >= 1:
                train_sequences.append(seq[:-1][-max_seq_len:])
        else:
            train_sequences.append(seq[-max_seq_len:])

    user_map = {user: idx + 1 for idx, user in enumerate(sorted(all_train_candidates.keys()))}

    # 5. Save input.txt and output.txt files
    print("Writing dataset files...")
    with open(output_dir / "train" / "input.txt", "w") as f:
        for seq in train_sequences:
            f.write(" ".join(map(str, seq)) + "\n")

    with open(output_dir / "val" / "input.txt", "w") as f:
        for seq in val_inputs:
            f.write(" ".join(map(str, seq)) + "\n")
    with open(output_dir / "val" / "output.txt", "w") as f:
        for target in val_outputs:
            f.write(str(target) + "\n")

    with open(output_dir / "test" / "input.txt", "w") as f:
        for seq in test_inputs:
            f.write(" ".join(map(str, seq)) + "\n")
    with open(output_dir / "test" / "output.txt", "w") as f:
        for target in test_outputs:
            f.write(str(target) + "\n")

    stats = {
        "num_users": len(user_map),
        "num_items": len(item_map),
        "num_train_sequences": len(train_sequences),
        "num_val_samples": len(val_inputs),
        "num_test_samples": len(test_inputs),
        "min_freq_warm": min_freq_warm,
        "min_user_core": min_user_core,
        "t_start": t_start,
        "t_split": t_split,
        "t_end": t_end,
        "t_start_iso": datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat(),
        "t_split_iso": datetime.fromtimestamp(t_split, tz=timezone.utc).isoformat(),
        "t_end_iso": datetime.fromtimestamp(t_end, tz=timezone.utc).isoformat()
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\n[DONE] Global Timeline Split successfully written to: {output_dir}")
    print(f"  • Items in catalog vocabulary: {stats['num_items']:,}")
    print(f"  • Training sequences:         {stats['num_train_sequences']:,}")
    print(f"  • Validation evaluation pairs: {stats['num_val_samples']:,}")
    print(f"  • Test evaluation pairs:       {stats['num_test_samples']:,}")

def main():
    parser = argparse.ArgumentParser(description="Global Timeline Split for Amazon 2023 (Petrov & Macdonald, RecSys '22)")
    parser.add_argument("--categories", nargs="+", default=["All_Beauty"],
                        help="List of Amazon 2023 categories to include or 'all' for all 34 categories.")
    parser.add_argument("--local_file", type=str, default=None,
                        help="Optional direct path to a local CSV or JSONL file (e.g. All_Beauty.jsonl/All_Beauty.jsonl).")
    parser.add_argument("--cache_dir", type=str, default="raw_amazon_2023",
                        help="Local cache directory for downloaded raw files.")
    parser.add_argument("--output_dir", type=str, default="datasets/amazon_gts",
                        help="Destination directory for processed dataset.")
    parser.add_argument("--target_warm_ratio", type=float, default=0.80,
                        help="Target fraction of warm items in test ground truth (default: 0.80).")
    parser.add_argument("--min_freq_warm", type=int, default=5,
                        help="Minimum training appearances to consider an item warm and include it in catalog (default: 5).")
    parser.add_argument("--min_user_core", type=int, default=2,
                        help="Minimum sequence length per user (e.g. 1 for 0-core, 2 for next-item, 5 for 5-core). Default: 2.")
    parser.add_argument("--val_users", type=int, default=1024,
                        help="Number of training users sampled for internal validation (default: 1024).")
    parser.add_argument("--test_days", type=int, default=7,
                        help="Duration of the test window in days (default: 7).")
    parser.add_argument("--t_end", type=float, default=None,
                        help="Optional fixed end timestamp. Defaults to max timestamp found in data.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine files to read
    files_to_process = []
    if args.local_file and os.path.exists(args.local_file):
        files_to_process.append(Path(args.local_file))
    else:
        cats = ALL_AMAZON_2023_CATEGORIES if "all" in args.categories else args.categories
        for c in cats:
            # Check if local exists first
            local_cand = Path(f"{c}.jsonl/{c}.jsonl")
            if local_cand.exists():
                files_to_process.append(local_cand)
            else:
                files_to_process.append(download_category_if_needed(c, cache_dir))

    # Determine global max timestamp (Amazon 2023 benchmark cutoff is 2023-09-09)
    # 1694227200.0 corresponds to 2023-09-09 00:00:00 UTC
    DEFAULT_AMAZON_2023_END = 1694227200.0
    t_end = args.t_end if args.t_end is not None else DEFAULT_AMAZON_2023_END
    print(f"Using reference end timestamp: {t_end} ({datetime.fromtimestamp(t_end, tz=timezone.utc)})")

    # Look back up to 180 days from t_end to keep memory footprint minimal during calibration
    candidate_window_start = t_end - (180 * 86400)
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

    # 3. Calibrate timeline (Single-Cut GTS)
    t_start, t_split, t_end, warm_ratio = calibrate_timeline(
        all_interactions,
        t_end=t_end,
        test_days=args.test_days,
        target_warm_ratio=args.target_warm_ratio,
        min_freq_warm=args.min_freq_warm
    )

    # 4. Export sequences in standard gSASRec / SASRec format
    build_sequences_and_export(
        all_interactions,
        t_start=t_start,
        t_split=t_split,
        t_end=t_end,
        output_dir=args.output_dir,
        val_users_count=args.val_users,
        min_freq_warm=args.min_freq_warm,
        min_user_core=args.min_user_core
    )

if __name__ == "__main__":
    main()
