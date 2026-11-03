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
import numpy as np

# Amazon Reviews 2023 Categories & URLs for backward compatibility
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


def download_file(url, target_path, retries=3):
    """
    Downloads a remote file with retries and temporary part file handling.
    """
    target_path = Path(target_path)
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Universal GTS Pipeline)"})
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


def download_amazon_category(category, cache_dir):
    """
    Downloads specified Amazon 2023 rating CSV file if not already cached.
    """
    cache_path = Path(cache_dir) / f"{category}.csv.gz"
    url = f"{RATING_BASE_URL}/{category}.csv.gz"
    return download_file(url, cache_path)


def normalize_timestamp(raw_val):
    """
    Normalizes different timestamp representations (ms, s, ISO string) to Unix epoch seconds (float).
    """
    if raw_val is None:
        return None
    if isinstance(raw_val, (int, float)):
        return raw_val / 1000.0 if raw_val > 1e11 else float(raw_val)
    
    val_str = str(raw_val).strip()
    if not val_str:
        return None
    try:
        t_float = float(val_str)
        return t_float / 1000.0 if t_float > 1e11 else t_float
    except ValueError:
        pass

    # Try ISO or standard datetime formats
    try:
        dt = datetime.fromisoformat(val_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(val_str, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            pass

    return None


def detect_column_indices(header, user_col=None, item_col=None, time_col=None):
    """
    Resolves column indices or names from a header list or dataframe columns.
    Uses user-specified names/indices or falls back to smart canonical heuristics.
    """
    header_lower = [str(c).strip().lower() for c in header]

    def resolve_col(specified, candidates, default_idx):
        if specified is not None:
            specified_str = str(specified).strip()
            if specified_str.isdigit():
                idx = int(specified_str)
                if idx < len(header):
                    return idx
            if specified_str.lower() in header_lower:
                return header_lower.index(specified_str.lower())
        for cand in candidates:
            if cand in header_lower:
                return header_lower.index(cand)
        return default_idx if default_idx < len(header) else 0

    user_idx = resolve_col(user_col, ["user_id", "userid", "reviewerid", "user", "uid"], 0)
    item_idx = resolve_col(item_col, ["item_id", "itemid", "parent_asin", "asin", "movieid", "item", "product_id", "iid"], 1)
    time_idx = resolve_col(time_col, ["timestamp", "time", "unixreviewtime", "timestamp_created", "date", "ts"], 3 if len(header) >= 4 else (2 if len(header) > 2 else 0))

    return user_idx, item_idx, time_idx


def stream_canonical_interactions(file_path, t_min=None, t_max=None, file_format=None, delimiter=None, user_col=None, item_col=None, time_col=None):
    """
    Streams canonical (user, item, timestamp) triples within [t_min, t_max]
    from CSV, TSV, DAT, JSONL, JSON.GZ, or PARQUET files.
    """
    path = Path(file_path)
    file_str = str(path).lower()
    interactions = []

    # Infer format if not explicitly provided
    if not file_format:
        if file_str.endswith(".parquet") or file_str.endswith(".pq"):
            file_format = "parquet"
        elif file_str.endswith(".jsonl") or file_str.endswith(".json") or file_str.endswith(".json.gz") or file_str.endswith(".jsonl.gz"):
            file_format = "jsonl"
        else:
            file_format = "delimited"

    # Infer delimiter
    if delimiter is None:
        if file_str.endswith(".tsv") or file_str.endswith(".tsv.gz"):
            delimiter = "\t"
        elif file_str.endswith(".dat"):
            # Check if MovieLens style '::' or tab or comma
            with open(path, "r", encoding="utf-8", errors="ignore") as f_peek:
                first_line = f_peek.readline()
                if "::" in first_line:
                    delimiter = "::"
                elif "\t" in first_line:
                    delimiter = "\t"
                else:
                    delimiter = ","
        else:
            delimiter = ","

    # PARQUET HANDLER (pyarrow / pandas)
    if file_format == "parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        u_col, i_col, t_col = detect_column_indices(list(df.columns), user_col, item_col, time_col)
        u_name = df.columns[u_col]
        i_name = df.columns[i_col]
        t_name = df.columns[t_col]

        # Extract columns
        for u_val, i_val, t_val in zip(df[u_name], df[i_name], df[t_name]):
            t = normalize_timestamp(t_val)
            if t is None:
                continue
            if (t_min is not None and t < t_min) or (t_max is not None and t > t_max):
                continue
            u = str(u_val).strip()
            i = str(i_val).strip()
            if u and i:
                interactions.append((u, i, t))
        return interactions

    # JSONL / JSON.GZ HANDLER
    if file_format == "jsonl":
        opener = gzip.open if (file_str.endswith(".gz")) else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Extract fields
                    u = data.get(user_col) if user_col else (data.get("user_id") or data.get("reviewerID") or data.get("userId"))
                    i = data.get(item_col) if item_col else (data.get("parent_asin") or data.get("asin") or data.get("item_id") or data.get("movieId"))
                    t_raw = data.get(time_col) if time_col else (data.get("timestamp") or data.get("unixReviewTime") or data.get("timestamp_created"))

                    t = normalize_timestamp(t_raw)
                    if t is None:
                        continue
                    if (t_min is not None and t < t_min) or (t_max is not None and t > t_max):
                        continue
                    if u and i:
                        interactions.append((str(u).strip(), str(i).strip(), t))
                except Exception:
                    continue
        return interactions

    # DELIMITED (CSV, TSV, DAT) HANDLER
    opener = gzip.open if file_str.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        if not first_line:
            return interactions

        # Check if first line is a header
        first_parts = first_line.strip().split(delimiter)
        has_header = any(col_name in first_line.lower() for col_name in ["user", "item", "time", "id", "asin", "movie", "rating"])
        
        if has_header:
            header = first_parts
            u_idx, i_idx, t_idx = detect_column_indices(header, user_col, item_col, time_col)
        else:
            # First line is already data
            header = [f"col_{idx}" for idx in range(len(first_parts))]
            u_idx, i_idx, t_idx = detect_column_indices(header, user_col, item_col, time_col)
            # Process first line as data
            if len(first_parts) > max(u_idx, i_idx, t_idx):
                t = normalize_timestamp(first_parts[t_idx])
                if t is not None and (t_min is None or t >= t_min) and (t_max is None or t <= t_max):
                    u = first_parts[u_idx].strip()
                    i = first_parts[i_idx].strip()
                    if u and i:
                        interactions.append((u, i, t))

        # Process remaining lines
        for line in f:
            parts = line.strip().split(delimiter)
            if len(parts) <= max(u_idx, i_idx, t_idx):
                continue
            t = normalize_timestamp(parts[t_idx])
            if t is None:
                continue
            if (t_min is not None and t < t_min) or (t_max is not None and t > t_max):
                continue
            u = parts[u_idx].strip()
            i = parts[i_idx].strip()
            if u and i:
                interactions.append((u, i, t))

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
        print(f"Iteration {iteration:2d}: {num_users_end:,} users, {len(post_item_counts):,} items (all items >= {min_freq_warm}: {all_items_warm})", flush=True)

        # Convergence condition: no users dropped and all items satisfy the warm threshold
        if num_users_start == num_users_end and all_items_warm:
            print(f"  [CONVERGED] Iterative {min_user_core}-core reached stability in {iteration} iteration(s).\n", flush=True)
            return new_users, set(post_item_counts.keys())

        current_users = new_users
        if len(current_users) == 0:
            print("WARNING: Iterative core filtering reduced training set to 0 users.", flush=True)
            return {}, set()


def export_canonical_metadata(warm_catalog, item_map, metadata_path, output_dir, meta_item_col=None, meta_title_col=None, meta_category_col=None,
                            delimiter=None):
    """
    Maps real-world product metadata (e.g. titles, genres, categories) from a CSV, TSV, DAT,
    or JSONL file to the newly created internal integer IDs [1..N].
    Exports:
      - {output_dir}/item_metadata.csv
      - {output_dir}/item_id_to_title.json
      - {output_dir}/item_id_to_orig_id.json
      - {output_dir}/orig_id_to_item_id.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orig_to_id = dict(item_map)
    id_to_orig = {v: k for k, v in item_map.items()}

    with open(output_dir / "orig_id_to_item_id.json", "w", encoding="utf-8") as f:
        json.dump(orig_to_id, f)
    with open(output_dir / "item_id_to_orig_id.json", "w", encoding="utf-8") as f:
        json.dump(id_to_orig, f)

    meta_path = Path(metadata_path) if metadata_path else None
    title_map = {}
    cat_map = {}

    if meta_path and meta_path.exists():
        print(f"\n[Metadata] Loading item metadata from: {meta_path}...", flush=True)
        file_str = str(meta_path).lower()

        if file_str.endswith(".jsonl") or file_str.endswith(".json"):
            with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        raw_id = str(d.get(meta_item_col) or d.get("item_id") or d.get("id") or d.get("parent_asin") or d.get("movieId") or "").strip()
                        if raw_id in warm_catalog:
                            t = d.get(meta_title_col) or d.get("title") or d.get("name") or f"Item {raw_id}"
                            c = d.get(meta_category_col) or d.get("category") or d.get("genre") or d.get("genres") or "General"
                            title_map[raw_id] = str(t).strip()
                            cat_map[raw_id] = str(c).strip()
                    except Exception:
                        continue
        else:
            # Delimited format
            delim = delimiter
            if delim is None:
                if "::" in open(meta_path, "r", encoding="utf-8", errors="ignore").readline():
                    delim = "::"
                elif file_str.endswith(".tsv"):
                    delim = "\t"
                else:
                    delim = ","

            with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                parts = first_line.strip().split(delim)
                has_hdr = any(w in first_line.lower() for w in ["id", "title", "name", "movie", "asin"])

                if has_hdr:
                    hdr_lower = [p.lower().strip() for p in parts]
                    i_idx = hdr_lower.index(meta_item_col.lower()) if meta_item_col and meta_item_col.lower() in hdr_lower else 0
                    t_idx = hdr_lower.index(meta_title_col.lower()) if meta_title_col and meta_title_col.lower() in hdr_lower else (1 if len(parts) > 1 else 0)
                    c_idx = hdr_lower.index(meta_category_col.lower()) if meta_category_col and meta_category_col.lower() in hdr_lower else (2 if len(parts) > 2 else -1)
                else:
                    i_idx, t_idx, c_idx = 0, (1 if len(parts) > 1 else 0), (2 if len(parts) > 2 else -1)
                    # Process first line as data
                    raw_id = parts[i_idx].strip()
                    if raw_id in warm_catalog:
                        title_map[raw_id] = parts[t_idx].strip() if len(parts) > t_idx else f"Item {raw_id}"
                        cat_map[raw_id] = parts[c_idx].strip() if c_idx >= 0 and len(parts) > c_idx else "General"

                for line in f:
                    parts = line.strip().split(delim)
                    if len(parts) <= max(i_idx, t_idx):
                        continue
                    raw_id = parts[i_idx].strip()
                    if raw_id in warm_catalog:
                        title_map[raw_id] = parts[t_idx].strip() if len(parts) > t_idx else f"Item {raw_id}"
                        cat_map[raw_id] = parts[c_idx].strip() if c_idx >= 0 and len(parts) > c_idx else "General"

        print(f"  Matched {len(title_map):,} out of {len(warm_catalog):,} warm items ({len(title_map)/max(1, len(warm_catalog))*100:.1f}%)", flush=True)

    # Export canonical CSV and JSON lookup
    csv_path = output_dir / "item_metadata.csv"
    id_to_title_path = output_dir / "item_id_to_title.json"
    id_to_title = {}

    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["item_id", "original_id", "title", "main_category"])
        for item_id in range(1, len(warm_catalog) + 1):
            orig_id = id_to_orig[item_id]
            title = title_map.get(orig_id, f"Item {orig_id}")
            cat = cat_map.get(orig_id, "General")
            writer.writerow([item_id, orig_id, title, cat])
            id_to_title[item_id] = title

    with open(id_to_title_path, "w", encoding="utf-8") as f_json:
        json.dump(id_to_title, f_json, indent=2)

    print(f"Exported: {csv_path}", flush=True)
    print(f"Exported: {id_to_title_path}", flush=True)


def build_and_export_splits(user_timeline, t_start, t_split, t_end, train_days, test_days, min_user_core, min_freq_warm, max_seq_len, val_users_count,
                            output_dir, metadata_path=None, meta_item_col=None, meta_title_col=None, meta_category_col=None, delimiter=None):
    """
    Executes the standard GTS splitting, k-core convergence, and dataset file serialization.
    Writes:
      - train/input.txt
      - val/input.txt and val/output.txt
      - test/input.txt and test/output.txt
      - dataset_stats.json
      - metadata mappings
    """
    output_dir = Path(output_dir)
    for sub in ["train", "val", "test"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # Filter training events strictly within [t_start, t_split)
    train_user_items = {}
    for u, history in user_timeline.items():
        items = [i for t, i in history if t_start <= t < t_split]
        if len(items) >= min_user_core:
            train_user_items[u] = items

    # Run iterative k-core decomposition on training window
    filtered_train_users, warm_catalog = run_iterative_k_core(train_user_items, min_user_core, min_freq_warm)
    if not filtered_train_users:
        raise RuntimeError("Core filtering resulted in 0 users. Relax min_user_core or min_freq_warm.")

    # Create vocabulary mappings (1-indexed for PyTorch Embedding)
    sorted_catalog = sorted(warm_catalog)
    item_map = {item: idx + 1 for idx, item in enumerate(sorted_catalog)}
    user_map = {u: idx + 1 for idx, u in enumerate(sorted(filtered_train_users.keys()))}

    print(f"Filtered Catalog Vocabulary: {len(item_map):,} warm items (all >= {min_freq_warm} in train)", flush=True)
    print(f"Surviving Training Users:     {len(user_map):,} users (all >= {min_user_core} warm items)", flush=True)

    # Build Test Set
    test_inputs, test_outputs = [], []
    for u, history in user_timeline.items():
        past_items = [item_map[i] for t, i in history if t_start <= t < t_split and i in item_map]
        future_events = [(t, i) for t, i in history if t_split < t <= t_end]

        if len(past_items) >= min_user_core and len(future_events) > 0:
            true_first_future_item = future_events[0][1]
            if true_first_future_item in item_map:
                test_inputs.append(past_items[-max_seq_len:])
                test_outputs.append(item_map[true_first_future_item])

    # Build Train and Validation Sets
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

    # Save text dataset files
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

    # Compute statistics and export dataset_stats.json
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
    print("=" * 80 + "\n", flush=True)

    # Export metadata
    export_canonical_metadata(
        warm_catalog=warm_catalog,
        item_map=item_map,
        metadata_path=metadata_path,
        output_dir=output_dir,
        meta_item_col=meta_item_col,
        meta_title_col=meta_title_col,
        meta_category_col=meta_category_col,
        delimiter=delimiter
    )

    return stats
