import os
import sys
import json
import gzip
import time
import shutil
import csv
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

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

RATING_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only"
META_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"


def download_file(url, target_path, retries=3, show_progress=False):
    """
    Downloads a remote file with retries and temporary part file handling.
    """
    target_path = Path(target_path)
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")

    if show_progress:
        print(f"  Downloading {url} ...", flush=True)

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Amazon GTS Dataset Builder)"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, "wb") as f_out:
                shutil.copyfileobj(resp, f_out)
            shutil.move(temp_path, target_path)
            return target_path
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            if attempt == retries:
                print(f"[ERROR] Failed to download {url} after {retries} attempts: {e}", flush=True)
                raise e
            time.sleep(2 * attempt)
    return target_path


def download_category_if_needed(category, cache_dir):
    """
    Downloads the specified Amazon 2023 rating CSV file if not already cached.
    """
    cache_path = cache_dir / f"{category}.csv.gz"
    url = f"{RATING_BASE_URL}/{category}.csv.gz"
    return download_file(url, cache_path)


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

    print(f"\nRunning iterative {min_user_core}-core decomposition on training window...", flush=True)
    while True:
        iteration += 1
        num_users_start = len(current_users)

        # Count item frequency
        item_counts = Counter()
        for items in current_users.values():
            for i in items:
                item_counts[i] += 1

        # Keep items meeting frequency threshold
        warm_items = {i for i, c in item_counts.items() if c >= min_freq_warm}

        # Filter users: keep only warm items and users with at least min_user_core items
        new_users = {}
        for u, items in current_users.items():
            filtered = [i for i in items if i in warm_items]
            if len(filtered) >= min_user_core:
                new_users[u] = filtered

        num_users_end = len(new_users)

        # Re-verify item frequencies on filtered users
        post_item_counts = Counter()
        for items in new_users.values():
            for i in items:
                post_item_counts[i] += 1

        all_items_warm = all(c >= min_freq_warm for c in post_item_counts.values()) if post_item_counts else False
        print(f"  • Iteration {iteration:2d}: {num_users_end:,} users, {len(post_item_counts):,} items (all items >= {min_freq_warm}: {all_items_warm})", flush=True)

        # Convergence condition: no users dropped and all items satisfy the warm threshold
        if num_users_start == num_users_end and all_items_warm:
            print(f"  [CONVERGED] Iterative {min_user_core}-core reached stability in {iteration} iteration(s).\n", flush=True)
            return new_users, set(post_item_counts.keys())

        current_users = new_users
        if len(current_users) == 0:
            print("WARNING: Iterative core filtering reduced training set to 0 users.", flush=True)
            return {}, set()


def export_item_metadata(
    warm_catalog,
    item_map,
    categories,
    cache_dir,
    output_dir,
    meta_threads=8,
    keep_meta_files=False
):
    """
    Downloads Amazon Reviews 2023 product metadata (meta_categories) and associates
    each warm item (parent_asin) with its real-world title, main category, brand/store,
    average rating, review count, and price.

    Exports:
      - {output_dir}/item_metadata.csv
      - {output_dir}/item_id_to_title.json
      - {output_dir}/item_id_to_asin.json
      - {output_dir}/asin_to_item_id.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_cache = Path(cache_dir) / "meta_categories"
    meta_cache.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("METADATA EXTRACTION: AMAZON REVIEWS 2023 (meta_categories)", flush=True)
    print(f"  Warm Items to Map : {len(warm_catalog):,}", flush=True)
    print(f"  Target Categories : {len(categories)}", flush=True)
    print(f"  Download Threads  : {meta_threads}", flush=True)
    print(f"  Keep Raw Files    : {keep_meta_files}", flush=True)
    print("=" * 80, flush=True)

    # 1. Export initial ID <-> ASIN mappings
    asin_to_id = dict(item_map)
    id_to_asin = {v: k for k, v in item_map.items()}

    with open(output_dir / "asin_to_item_id.json", "w", encoding="utf-8") as f:
        json.dump(asin_to_id, f)
    with open(output_dir / "item_id_to_asin.json", "w", encoding="utf-8") as f:
        json.dump(id_to_asin, f)
    print(f"  • Exported: {output_dir / 'asin_to_item_id.json'}", flush=True)
    print(f"  • Exported: {output_dir / 'item_id_to_asin.json'}", flush=True)

    # 2. Worker function to download and scan a single category metadata file
    metadata_map = {}

    def process_metadata_category(cat_name):
        url = f"{META_BASE_URL}/meta_{cat_name}.jsonl.gz"
        dest = meta_cache / f"meta_{cat_name}.jsonl.gz"
        download_file(url, dest)

        found_in_cat = {}
        with gzip.open(dest, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    asin = data.get("parent_asin")
                    if asin in warm_catalog and asin not in found_in_cat:
                        title = (data.get("title") or "").strip().replace("\n", " ").replace("\r", " ")
                        main_cat = data.get("main_category") or cat_name
                        store = data.get("store") or ""
                        avg_rating = data.get("average_rating") or ""
                        rating_num = data.get("rating_number") or ""
                        price = data.get("price") or ""

                        found_in_cat[asin] = {
                            "title": title,
                            "main_category": main_cat,
                            "store": store,
                            "average_rating": avg_rating,
                            "rating_number": rating_num,
                            "price": price
                        }
                except Exception:
                    continue

        # Clean up raw gz file immediately to save disk quota on cloud instances
        if not keep_meta_files and dest.exists():
            dest.unlink()

        return cat_name, found_in_cat

    # 3. Process categories in parallel
    print(f"\nStreaming metadata across {len(categories)} categories (threads={meta_threads})...", flush=True)
    with ThreadPoolExecutor(max_workers=min(meta_threads, len(categories))) as executor:
        futures = {executor.submit(process_metadata_category, c): c for c in categories}
        for future in as_completed(futures):
            cat = futures[future]
            try:
                category, found = future.result()
                metadata_map.update(found)
                pct = (len(metadata_map) / max(1, len(warm_catalog))) * 100
                print(f"  • {category:30}: {len(found):5,} items found (Total: {len(metadata_map):,}/{len(warm_catalog):,} - {pct:.1f}%)", flush=True)
            except Exception as e:
                print(f"  [WARNING] Metadata extraction failed for {cat}: {e}", flush=True)

    # 4. Export CSV and JSON title lookup
    csv_path = output_dir / "item_metadata.csv"
    id_to_title_path = output_dir / "item_id_to_title.json"
    id_to_title = {}

    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["item_id", "parent_asin", "title", "main_category", "store", "average_rating", "rating_number", "price"])

        for item_id in range(1, len(warm_catalog) + 1):
            asin = id_to_asin[item_id]
            meta = metadata_map.get(asin, {})
            title = meta.get("title") or f"Item {asin}"
            main_cat = meta.get("main_category") or "Unknown"
            store = meta.get("store") or ""
            avg_rating = meta.get("average_rating") or ""
            rating_num = meta.get("rating_number") or ""
            price = meta.get("price") or ""

            writer.writerow([item_id, asin, title, main_cat, store, avg_rating, rating_num, price])
            id_to_title[item_id] = title

    with open(id_to_title_path, "w", encoding="utf-8") as f_json:
        json.dump(id_to_title, f_json)

    print("\n" + "=" * 80, flush=True)
    print("METADATA EXPORT SUMMARY:", flush=True)
    print(f"  • CSV file generated  : {csv_path} ({csv_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)
    print(f"  • JSON titles file    : {id_to_title_path} ({id_to_title_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)
    print(f"  • Titles coverage     : {len(metadata_map):,} / {len(warm_catalog):,} ({(len(metadata_map)/max(1, len(warm_catalog)))*100:.2f}%)", flush=True)
    print("=" * 80 + "\n", flush=True)

    # 5. Preview decoded test sequences if test files exist
    test_inp_path = output_dir / "test" / "input.txt"
    test_out_path = output_dir / "test" / "output.txt"
    if test_inp_path.exists() and test_out_path.exists():
        print("--- PREVIEW: REAL TEST SEQUENCES WITH PRODUCT TITLES ---", flush=True)
        with open(test_inp_path, "r", encoding="utf-8") as f_in, open(test_out_path, "r", encoding="utf-8") as f_out:
            for sample_idx in range(2):
                inp_line = f_in.readline().strip()
                out_line = f_out.readline().strip()
                if not inp_line or not out_line:
                    break
                seq_ids = [int(x) for x in inp_line.split()]
                target_id = int(out_line)

                print(f"\n[Test User Sample #{sample_idx + 1}]", flush=True)
                print("  Purchase History (Transformer Input):", flush=True)
                for pos, i_id in enumerate(seq_ids[-5:], start=max(1, len(seq_ids) - 4)):
                    print(f"    {pos}. [ID: {i_id:6d}] {id_to_title.get(i_id, 'N/A')[:80]}", flush=True)
                print(f"  -> Next-Item Target: [ID: {target_id:6d}] {id_to_title.get(target_id, 'N/A')[:80]}", flush=True)
        print("=" * 80 + "\n", flush=True)


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
    output_dir,
    export_metadata=False,
    only_metadata=False,
    categories=None,
    cache_dir="raw_amazon_2023",
    meta_threads=8,
    keep_meta_files=False
):
    output_dir = Path(output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)

    # Extract raw training interactions strictly in [t_start, t_split)
    train_user_items = {}
    for u, history in user_timeline.items():
        train_items = [i for t, i in history if t_start <= t < t_split]
        if len(train_items) >= min_user_core:
            train_user_items[u] = train_items

    print("="*80, flush=True)
    print("DATASET PREPARATION: GLOBAL TIMELINE SPLIT + ITERATIVE K-CORE", flush=True)
    print(f"  Training Window  : [{datetime.fromtimestamp(t_start, tz=timezone.utc)} -> {datetime.fromtimestamp(t_split, tz=timezone.utc)}) ({train_days} days)", flush=True)
    print(f"  Test Window      : ({datetime.fromtimestamp(t_split, tz=timezone.utc)} -> {datetime.fromtimestamp(t_end, tz=timezone.utc)}] ({test_days} days)", flush=True)
    print(f"  Min User Core    : >= {min_user_core}", flush=True)
    print(f"  Min Warm Freq    : >= {min_freq_warm}", flush=True)
    print(f"  Export Metadata  : {export_metadata}", flush=True)
    print("="*80, flush=True)

    # Run Iterative k-core decomposition on train
    filtered_train_users, warm_catalog = run_iterative_k_core(train_user_items, min_user_core, min_freq_warm)

    if not filtered_train_users or not warm_catalog:
        raise RuntimeError("No users or items survived the iterative k-core filtering.")

    # 3. Create 1-indexed contiguous item mapping
    sorted_catalog = sorted(warm_catalog)
    item_map = {item: idx + 1 for idx, item in enumerate(sorted_catalog)}
    user_map = {u: idx + 1 for idx, u in enumerate(sorted(filtered_train_users.keys()))}

    print(f"Filtered Catalog Vocabulary: {len(item_map):,} warm items (all >= {min_freq_warm} in train)", flush=True)
    print(f"Surviving Training Users:     {len(user_map):,} users (all >= {min_user_core} warm items)", flush=True)

    # If --only_metadata is specified, skip dataset file generation and run metadata export directly
    if only_metadata:
        print("\n[--only_metadata active] Skipping train/val/test file generation.", flush=True)
        cats_to_fetch = ALL_AMAZON_2023_CATEGORIES if not categories or "all" in categories else categories
        export_item_metadata(
            warm_catalog=warm_catalog,
            item_map=item_map,
            categories=cats_to_fetch,
            cache_dir=cache_dir,
            output_dir=output_dir,
            meta_threads=meta_threads,
            keep_meta_files=keep_meta_files
        )
        return

    # 4. Build Test Set
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
    rng = np.random.RandomState(42)
    eligible_val_users = [u for u, items in filtered_train_users.items() if len(items) >= 2]
    num_val = min(val_users_count, len(eligible_val_users))
    val_user_set = set(rng.choice(eligible_val_users, num_val, replace=False)) if num_val > 0 else set()

    train_sequences = []
    val_inputs, val_outputs = [], []

    for u, items in filtered_train_users.items():
        seq = [item_map[i] for i in items]
        if u in val_user_set:
            val_inputs.append(seq[:-1][-max_seq_len:])
            val_outputs.append(seq[-1])
            if len(seq[:-1]) >= 1:
                train_sequences.append(seq[:-1][-max_seq_len:])
        else:
            train_sequences.append(seq[-max_seq_len:])

    # 6. Save text dataset files
    print("\nWriting dataset files...", flush=True)
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

    print(f"\n[DONE] Global Timeline Split successfully written to: {output_dir}", flush=True)
    print(f"  • Items in catalog vocabulary: {stats['num_items']:,}", flush=True)
    print(f"  • Training sequences:         {stats['num_train_sequences']:,}", flush=True)
    print(f"  • Training interactions:      {stats['num_train_interactions']:,} (avg length: {stats['avg_train_seq_len']})", flush=True)
    print(f"  • Validation evaluation samples: {stats['num_val_samples']:,}", flush=True)
    print(f"  • Test evaluation samples:       {stats['num_test_samples']:,}", flush=True)
    print("="*80 + "\n", flush=True)

    # 8. Export metadata if requested
    if export_metadata:
        cats_to_fetch = ALL_AMAZON_2023_CATEGORIES if not categories or "all" in categories else categories
        export_item_metadata(
            warm_catalog=warm_catalog,
            item_map=item_map,
            categories=cats_to_fetch,
            cache_dir=cache_dir,
            output_dir=output_dir,
            meta_threads=meta_threads,
            keep_meta_files=keep_meta_files
        )


def main():
    parser = argparse.ArgumentParser(description="Build and Export Amazon 2023 GTS Dataset with Iterative Core Filtering & Metadata")
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
    parser.add_argument("--export_metadata", action="store_true",
                        help="Download product metadata and export item_metadata.csv, item_id_to_title.json, etc.")
    parser.add_argument("--only_metadata", action="store_true",
                        help="Reconstruct warm catalog and export metadata only without re-generating train/val/test splits.")
    parser.add_argument("--meta_threads", type=int, default=8,
                        help="Number of parallel worker threads for downloading metadata files (default: 8).")
    parser.add_argument("--keep_meta_files", action="store_true",
                        help="Keep raw meta_{category}.jsonl.gz files on disk after processing (default: False, cleans up to save disk space).")
    args = parser.parse_args()

    if args.only_metadata:
        args.export_metadata = True

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    DEFAULT_AMAZON_2023_END = 1694227200.0
    t_end = args.t_end if args.t_end is not None else DEFAULT_AMAZON_2023_END
    t_split = t_end - (args.test_days * 86400)
    t_start = t_split - (args.train_days * 86400)

    # Print startup banner
    print("=" * 80, flush=True)
    print("AMAZON 2023 GLOBAL TIMELINE SPLIT DATASET BUILDER", flush=True)
    print(f"  Reference End Timestamp : {t_end} ({datetime.fromtimestamp(t_end, tz=timezone.utc)})", flush=True)
    print(f"  Global Split Timestamp  : {t_split} ({datetime.fromtimestamp(t_split, tz=timezone.utc)})", flush=True)
    print(f"  Training Start Timestamp: {t_start} ({datetime.fromtimestamp(t_start, tz=timezone.utc)})", flush=True)
    print(f"  Train Days              : {args.train_days}", flush=True)
    print(f"  Test Days               : {args.test_days}", flush=True)
    print(f"  Only Metadata Mode      : {args.only_metadata}", flush=True)
    print(f"  Export Metadata         : {args.export_metadata}", flush=True)
    print("=" * 80, flush=True)

    # Download / prepare candidate rating files
    print("\n[Step 1] Preparing source rating files...", flush=True)
    files_to_process = []
    if args.local_file and os.path.exists(args.local_file):
        files_to_process.append(Path(args.local_file))
        print(f"  Using local file: {args.local_file}", flush=True)
    else:
        cats = ALL_AMAZON_2023_CATEGORIES if "all" in args.categories else args.categories
        to_download = []
        for c in cats:
            local_cand = Path(f"{c}.jsonl/{c}.jsonl")
            if local_cand.exists():
                files_to_process.append(local_cand)
                print(f"  [Local JSONL] Found {local_cand}", flush=True)
            else:
                cache_path = cache_dir / f"{c}.csv.gz"
                if cache_path.exists() and cache_path.stat().st_size > 0:
                    files_to_process.append(cache_path)
                    print(f"  [Cached] {cache_path.name} ({cache_path.stat().st_size / (1024*1024):.1f} MB)", flush=True)
                else:
                    to_download.append(c)

        if to_download:
            print(f"  Downloading {len(to_download)} rating category files in parallel (threads={args.meta_threads})...", flush=True)
            with ThreadPoolExecutor(max_workers=args.meta_threads) as executor:
                future_to_cat = {executor.submit(download_category_if_needed, c, cache_dir): c for c in to_download}
                for future in as_completed(future_to_cat):
                    c = future_to_cat[future]
                    try:
                        dest = future.result()
                        files_to_process.append(dest)
                        sz_mb = dest.stat().st_size / (1024 * 1024)
                        print(f"  [OK] Downloaded {dest.name:30} ({sz_mb:6.1f} MB)", flush=True)
                    except Exception as e:
                        print(f"  [ERROR] Failed to download {c}: {e}", flush=True)
                        raise e

    # Load candidate interactions strictly in [t_start, t_end]
    print(f"\n[Step 2] Loading interactions from {len(files_to_process)} source(s) in window [{t_start}, {t_end}]...", flush=True)
    all_interactions = []
    t0 = time.time()
    for idx, f in enumerate(files_to_process, 1):
        print(f"  [{idx}/{len(files_to_process)}] Reading from {f.name}...", flush=True)
        sub = stream_interactions_in_window(f, t_start, t_end)
        all_interactions.extend(sub)
        print(f"      Extracted {len(sub):,} interactions within window.", flush=True)

    if not all_interactions:
        print("ERROR: No interactions found within the specified time window!", flush=True)
        return

    print(f"\nTotal interactions extracted: {len(all_interactions):,} in {time.time() - t0:.1f}s", flush=True)

    # Chronological sort and deduplication per user
    print(f"\n[Step 3] Sorting and deduplicating {len(all_interactions):,} interactions...", flush=True)
    user_timeline = defaultdict(list)
    seen = set()
    for u, i, t in all_interactions:
        key = (u, i, t)
        if key not in seen:
            seen.add(key)
            user_timeline[u].append((t, i))

    for u in user_timeline:
        user_timeline[u].sort(key=lambda x: x[0])

    print(f"Retained {sum(len(h) for h in user_timeline.values()):,} unique interactions across {len(user_timeline):,} users.\n", flush=True)

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
        output_dir=args.output_dir,
        export_metadata=args.export_metadata,
        only_metadata=args.only_metadata,
        categories=args.categories,
        cache_dir=args.cache_dir,
        meta_threads=args.meta_threads,
        keep_meta_files=args.keep_meta_files
    )


if __name__ == "__main__":
    main()
