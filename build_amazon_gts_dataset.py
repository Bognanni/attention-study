import os
import json
import gzip
import shutil
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import argparse
import numpy as np

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

def run_iterative_k_core(train_user_items, min_user_core, min_freq_warm):
    """
    Applies iterative k-core decomposition strictly on the training partition.
    Alternates item filtering and user filtering until full convergence.
    """
    iteration = 0
    current_users = dict(train_user_items)

    print(f"\nRunning iterative {min_user_core}-core decomposition on training window...")
    while True:
        iteration += 1
        num_users_start = len(current_users)

        # 1. Count item frequency
        item_counts = Counter()
        for items in current_users.values():
            for i in items:
                item_counts[i] += 1

        # Keep items meeting frequency threshold
        warm_items = {i for i, c in item_counts.items() if c >= min_freq_warm}

        # 2. Filter users: keep only warm items and users with at least min_user_core items
        new_users = {}
        for u, items in current_users.items():
            filtered = [i for i in items if i in warm_items]
            if len(filtered) >= min_user_core:
                new_users[u] = filtered

        num_users_end = len(new_users)

        # 3. Re-verify item frequencies on filtered users
        post_item_counts = Counter()
        for items in new_users.values():
            for i in items:
                post_item_counts[i] += 1

        all_items_warm = all(c >= min_freq_warm for c in post_item_counts.values())
        print(f"  • Iteration {iteration}: {num_users_end:,} users, {len(post_item_counts):,} items (all items >= {min_freq_warm}: {all_items_warm})")

        # Convergence condition: no users dropped and all items satisfy the warm threshold
        if num_users_start == num_users_end and all_items_warm:
            print(f"  [CONVERGED] Iterative {min_user_core}-core reached stability in {iteration} iteration(s).\n")
            return new_users, set(post_item_counts.keys())

        current_users = new_users
        if len(current_users) == 0:
            print(f"WARNING: Iterative core filtering reduced training set to 0 users.")
            return {}, set()

def build_and_export_dataset(
    user_timeline,
    t_start,
    t_split,
    t_end,
    train_days,
    test_days,
    min_user_core,
    min_freq_warm,
    max_seq_len,
    val_users_count,
    output_dir
):
    output_dir = Path(output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)

    # 1. Extract raw training interactions strictly in [t_start, t_split)
    train_user_items = {}
    for u, history in user_timeline.items():
        train_items = [i for t, i in history if t_start <= t < t_split]
        if len(train_items) >= min_user_core:
            train_user_items[u] = train_items

    print("="*80)
    print("DATASET PREPARATION: GLOBAL TIMELINE SPLIT + ITERATIVE K-CORE")
    print(f"  Training Window  : [{datetime.fromtimestamp(t_start, tz=timezone.utc)} -> {datetime.fromtimestamp(t_split, tz=timezone.utc)}) ({train_days} days)")
    print(f"  Test Window      : ({datetime.fromtimestamp(t_split, tz=timezone.utc)} -> {datetime.fromtimestamp(t_end, tz=timezone.utc)}] ({test_days} days)")
    print(f"  Min User Core    : >= {min_user_core}")
    print(f"  Min Warm Freq    : >= {min_freq_warm}")
    print("="*80)

    # 2. Run Iterative k-core decomposition on train
    filtered_train_users, warm_catalog = run_iterative_k_core(train_user_items, min_user_core, min_freq_warm)

    if not filtered_train_users or not warm_catalog:
        raise RuntimeError("No users or items survived the iterative k-core filtering.")

    # 3. Create 1-indexed contiguous item mapping
    sorted_catalog = sorted(warm_catalog)
    item_map = {item: idx + 1 for idx, item in enumerate(sorted_catalog)}
    user_map = {u: idx + 1 for idx, u in enumerate(sorted(filtered_train_users.keys()))}

    print(f"Filtered Catalog Vocabulary: {len(item_map):,} warm items (all >= {min_freq_warm} in train)")
    print(f"Surviving Training Users:     {len(user_map):,} users (all >= {min_user_core} warm items)")

    # 4. Build Test Set
    # Input history strictly in [t_start, t_split) using item_map; Target is the first item post-t_split
    test_inputs, test_outputs = [], []
    for u, history in user_timeline.items():
        past_items = [item_map[i] for t, i in history if t_start <= t < t_split and i in item_map]
        future_events = [(t, i) for t, i in history if t_split < t <= t_end]

        if len(past_items) >= min_user_core and len(future_events) > 0:
            true_first_future_item = future_events[0][1]
            if true_first_future_item in item_map:
                test_inputs.append(past_items[-max_seq_len:])
                test_outputs.append(item_map[true_first_future_item])

    # 5. Build Train & Validation Sets
    # Select validation users from filtered training users
    rng = np.random.RandomState(42)
    eligible_val_users = [u for u, items in filtered_train_users.items() if len(items) >= 2]
    num_val = min(val_users_count, len(eligible_val_users))
    val_user_set = set(rng.choice(eligible_val_users, num_val, replace=False)) if num_val > 0 else set()

    train_sequences = []
    val_inputs, val_outputs = [], []

    for u, items in filtered_train_users.items():
        seq = [item_map[i] for i in items]
        if u in val_user_set:
            # Leave the last item of train for validation
            val_inputs.append(seq[:-1][-max_seq_len:])
            val_outputs.append(seq[-1])
            # Train sequence gets the remaining prefix
            if len(seq[:-1]) >= 1:
                train_sequences.append(seq[:-1][-max_seq_len:])
        else:
            train_sequences.append(seq[-max_seq_len:])

    # 6. Save text dataset files
    print("\nWriting dataset files...")
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

    # 7. Compute statistics and export dataset_stats.json
    num_train_interactions = sum(len(seq) for seq in train_sequences)
    avg_train_seq_len = round(num_train_interactions / max(1, len(train_sequences)), 2)

    stats = {
        "num_users": len(user_map),
        "num_items": len(item_map),
        "num_train_sequences": len(train_sequences),
        "num_train_interactions": num_train_interactions,
        "avg_train_seq_len": avg_train_seq_len,
        "num_val_samples": len(val_inputs),
        "num_test_samples": len(test_inputs),
        "train_days": train_days,
        "test_days": test_days,
        "min_freq_warm": min_freq_warm,
        "min_user_core": min_user_core,
        "max_seq_len": max_seq_len,
        "iterative_core_filtering": True,
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
    print(f"  • Training interactions:      {stats['num_train_interactions']:,} (avg length: {stats['avg_train_seq_len']})")
    print(f"  • Validation evaluation samples: {stats['num_val_samples']:,}")
    print(f"  • Test evaluation samples:       {stats['num_test_samples']:,}")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Build and Export Amazon 2023 GTS Dataset with Iterative Core Filtering")
    parser.add_argument("--categories", nargs="+", default=["All_Beauty"],
                        help="List of Amazon 2023 categories to include or 'all' for all 34 categories.")
    parser.add_argument("--local_file", type=str, default=None,
                        help="Optional direct path to a local CSV or JSONL file (e.g. All_Beauty.jsonl/All_Beauty.jsonl).")
    parser.add_argument("--cache_dir", type=str, default="raw_amazon_2023",
                        help="Local cache directory for downloaded raw files.")
    parser.add_argument("--output_dir", type=str, default="datasets/amazon_gts",
                        help="Destination directory for processed dataset.")
    parser.add_argument("--train_days", type=int, default=120,
                        help="Duration of the training window in days (default: 120).")
    parser.add_argument("--test_days", type=int, default=15,
                        help="Duration of the test window in days (default: 15).")
    parser.add_argument("--min_user_core", type=int, default=5,
                        help="Minimum sequence length per user in training window (default: 5).")
    parser.add_argument("--min_freq_warm", type=int, default=5,
                        help="Minimum training appearances to consider an item warm and include in catalog (default: 5).")
    parser.add_argument("--max_seq_len", type=int, default=50,
                        help="Maximum sequence length truncation for Transformer (default: 50).")
    parser.add_argument("--val_users", type=int, default=1024,
                        help="Number of training users sampled for internal validation (default: 1024).")
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
    t_split = t_end - (args.test_days * 86400)
    t_start = t_split - (args.train_days * 86400)

    print(f"Reference End Timestamp : {t_end} ({datetime.fromtimestamp(t_end, tz=timezone.utc)})")
    print(f"Global Split Timestamp  : {t_split} ({datetime.fromtimestamp(t_split, tz=timezone.utc)})")
    print(f"Training Start Timestamp: {t_start} ({datetime.fromtimestamp(t_start, tz=timezone.utc)})")

    # Load candidate interactions strictly in [t_start, t_end]
    print(f"\nLoading interactions from {len(files_to_process)} source(s) in window [{t_start}, {t_end}]...")
    all_interactions = []
    for f in files_to_process:
        print(f"Reading from {f.name}...")
        sub = stream_interactions_in_window(f, t_start, t_end)
        all_interactions.extend(sub)
        print(f"  Extracted {len(sub):,} interactions within window.")

    if not all_interactions:
        print("ERROR: No interactions found within the specified time window!")
        return

    # Chronological sort and deduplication per user
    print(f"Sorting and deduplicating {len(all_interactions):,} interactions...")
    user_timeline = defaultdict(list)
    seen = set()
    for u, i, t in all_interactions:
        key = (u, i, t)
        if key not in seen:
            seen.add(key)
            user_timeline[u].append((t, i))

    for u in user_timeline:
        user_timeline[u].sort(key=lambda x: x[0])

    print(f"Retained {sum(len(h) for h in user_timeline.values()):,} unique interactions across {len(user_timeline):,} users.")

    build_and_export_dataset(
        user_timeline=user_timeline,
        t_start=t_start,
        t_split=t_split,
        t_end=t_end,
        train_days=args.train_days,
        test_days=args.test_days,
        min_user_core=args.min_user_core,
        min_freq_warm=args.min_freq_warm,
        max_seq_len=args.max_seq_len,
        val_users_count=args.val_users,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
