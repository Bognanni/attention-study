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
from collections import defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from gts_core import (
    stream_canonical_interactions,
    build_and_export_splits,
    ALL_AMAZON_2023_CATEGORIES,
    RATING_BASE_URL,
    META_BASE_URL,
    download_file,
    download_amazon_category
)


def export_amazon_metadata( warm_catalog, item_map, categories, cache_dir, output_dir, meta_threads=8, keep_meta_files=False):
    """
    Downloads Amazon Reviews 2023 product metadata (meta_categories) and associates
    each warm item (parent_asin) with its real-world title, main category, brand/store, etc.
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
    print("=" * 80, flush=True)

    asin_to_id = dict(item_map)
    id_to_asin = {v: k for k, v in item_map.items()}

    with open(output_dir / "asin_to_item_id.json", "w", encoding="utf-8") as f:
        json.dump(asin_to_id, f)
    with open(output_dir / "item_id_to_asin.json", "w", encoding="utf-8") as f:
        json.dump(id_to_asin, f)
    with open(output_dir / "orig_id_to_item_id.json", "w", encoding="utf-8") as f:
        json.dump(asin_to_id, f)
    with open(output_dir / "item_id_to_orig_id.json", "w", encoding="utf-8") as f:
        json.dump(id_to_asin, f)

    metadata_map = {}

    def process_cat(cat_name):
        """
        Downloads and processes a single Amazon category metadata file.
        """
        url = f"{META_BASE_URL}/meta_{cat_name}.jsonl.gz"
        dest = meta_cache / f"meta_{cat_name}.jsonl.gz"
        download_file(url, dest)

        found = {}
        with gzip.open(dest, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    asin = data.get("parent_asin")
                    if asin in warm_catalog and asin not in found:
                        found[asin] = {
                            "title": (data.get("title") or "").strip().replace("\n", " ").replace("\r", " "),
                            "main_category": data.get("main_category") or cat_name,
                            "store": data.get("store") or "",
                            "average_rating": data.get("average_rating") or "",
                            "rating_number": data.get("rating_number") or "",
                            "price": data.get("price") or ""
                        }
                except Exception:
                    continue

        if not keep_meta_files and dest.exists():
            dest.unlink()
        return cat_name, found

    with ThreadPoolExecutor(max_workers=min(meta_threads, len(categories))) as executor:
        futures = {executor.submit(process_cat, c): c for c in categories}
        for future in as_completed(futures):
            cat = futures[future]
            try:
                category, found = future.result()
                metadata_map.update(found)
                pct = (len(metadata_map) / max(1, len(warm_catalog))) * 100
                print(f"  • {category:30}: {len(found):5,} items found (Total: {len(metadata_map):,}/{len(warm_catalog):,} - {pct:.1f}%)", flush=True)
            except Exception as e:
                print(f"  [WARNING] Metadata extraction failed for {cat}: {e}", flush=True)

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
        json.dump(id_to_title, f_json, indent=2)

    print(f"Exported: {csv_path}", flush=True)
    print(f"Exported: {id_to_title_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Universal Global Timeline Split (GTS) Dataset Builder")
    # Dataset Identity
    parser.add_argument("--dataset_name", type=str, default="custom_gts",
                        help="Identifier name for dataset (e.g. movielens_1m, steam, amazon_gts).")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to raw dataset file (CSV, TSV, DAT, JSONL, PARQUET) or directory.")
    parser.add_argument("--format", type=str, default=None, choices=["csv", "tsv", "dat", "jsonl", "parquet"],
                        help="Format of data files. If omitted, inferred from file extensions.")
    parser.add_argument("--delimiter", type=str, default=None,
                        help="Delimiter for text/dat files (e.g. ',' or '::' or '\\t').")
    parser.add_argument("--user_col", type=str, default=None,
                        help="Name or 0-based index of user ID column.")
    parser.add_argument("--item_col", type=str, default=None,
                        help="Name or 0-based index of item ID column.")
    parser.add_argument("--time_col", type=str, default=None,
                        help="Name or 0-based index of timestamp column.")
    # External Metadata Options
    parser.add_argument("--metadata_path", type=str, default=None,
                        help="Path to external item metadata file (e.g. movies.dat, games.jsonl).")
    parser.add_argument("--meta_item_col", type=str, default=None,
                        help="Column name or index for item ID in metadata file.")
    parser.add_argument("--meta_title_col", type=str, default=None,
                        help="Column name or index for product/movie title in metadata file.")
    parser.add_argument("--meta_category_col", type=str, default=None,
                        help="Column name or index for category/genre in metadata file.")
    # Output Destination
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Destination directory for processed dataset. Defaults to datasets/<dataset_name>.")
    # GTS Slicing Parameters
    parser.add_argument("--train_days", type=int, default=120,
                        help="Duration of training window in days (default: 120).")
    parser.add_argument("--test_days", type=int, default=15,
                        help="Duration of test window in days (default: 15).")
    parser.add_argument("--t_start", type=float, default=None,
                        help="Explicit start timestamp. Overrides train_days calculation if provided.")
    parser.add_argument("--t_split", type=float, default=None,
                        help="Explicit split timestamp. Overrides test_days calculation if provided.")
    parser.add_argument("--t_end", type=float, default=None,
                        help="Explicit end timestamp. Defaults to max timestamp observed.")
    # Sequential Recommendation Constraints
    parser.add_argument("--min_user_core", type=int, default=5,
                        help="Minimum sequence length per user in training window (default: 5).")
    parser.add_argument("--min_freq_warm", type=int, default=5,
                        help="Minimum training appearances to consider an item warm (default: 5).")
    parser.add_argument("--max_seq_len", type=int, default=50,
                        help="Maximum sequence length truncation for Transformer (default: 50).")
    parser.add_argument("--val_users", type=int, default=1024,
                        help="Number of training users sampled for internal validation (default: 1024).")
    # Amazon 2023 Specific Mode (backward compatibility)
    parser.add_argument("--categories", nargs="+", default=None,
                        help="List of Amazon 2023 categories or 'all'. Activates Amazon mode if specified.")
    parser.add_argument("--cache_dir", type=str, default="raw_amazon_2023",
                        help="Local cache directory for downloaded Amazon files.")
    parser.add_argument("--export_metadata", action="store_true",
                        help="Download product metadata and export item_metadata.csv, item_id_to_title.json, etc.")
    parser.add_argument("--only_metadata", action="store_true",
                        help="Reconstruct warm catalog and export metadata only without re-generating splits.")
    parser.add_argument("--meta_threads", type=int, default=8,
                        help="Number of parallel worker threads for downloading metadata files (default: 8).")
    parser.add_argument("--keep_meta_files", action="store_true",
                        help="Keep raw meta files on disk after processing.")
    args = parser.parse_args()

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"datasets/{args.dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve source files
    files_to_process = []
    is_amazon_mode = bool(args.categories)

    if is_amazon_mode:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cats = ALL_AMAZON_2023_CATEGORIES if "all" in args.categories else args.categories
        to_download = []
        for c in cats:
            local_cand = Path(f"{c}.jsonl/{c}.jsonl")
            if local_cand.exists():
                files_to_process.append(local_cand)
                print(f"  [Local JSONL] Found {local_cand}", flush=True)
            else:
                cp = cache_dir / f"{c}.csv.gz"
                if cp.exists() and cp.stat().st_size > 0:
                    files_to_process.append(cp)
                else:
                    to_download.append(c)

        if to_download:
            print(f"  Downloading {len(to_download)} Amazon category files (threads={args.meta_threads})...", flush=True)
            with ThreadPoolExecutor(max_workers=args.meta_threads) as executor:
                futures = {executor.submit(download_amazon_category, c, cache_dir): c for c in to_download}
                for f in as_completed(futures):
                    files_to_process.append(f.result())
    elif args.data_path:
        p = Path(args.data_path)
        if p.is_dir():
            files_to_process.extend(sorted(p.glob("*.*")))
        elif p.exists():
            files_to_process.append(p)
        else:
            raise FileNotFoundError(f"Data path not found: {args.data_path}")
    else:
        parser.print_help()
        print("\n[ERROR] No dataset source specified! Please provide either --data_path <file/folder> or --categories <cat1> ...", flush=True)
        return

    # First pass: observe timestamp range if t_end/t_split not provided
    print(f"\n[Step 1] Scanning {len(files_to_process)} source file(s)...", flush=True)
    all_interactions = []
    min_observed_t = float("inf")
    max_observed_t = float("-inf")
    t0 = time.time()

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
            min_observed_t = min(min_observed_t, min(t_vals))
            max_observed_t = max(max_observed_t, max(t_vals))
            all_interactions.extend(sub)
        print(f"      Extracted {len(sub):,} interactions.", flush=True)

    if not all_interactions:
        print("ERROR: No valid interactions extracted from source files.")
        return

    print(f"Extracted {len(all_interactions):,} interactions in {time.time() - t0:.1f}s")
    print(f"Observed Time Range: {datetime.fromtimestamp(min_observed_t, tz=timezone.utc)} to {datetime.fromtimestamp(max_observed_t, tz=timezone.utc)}")

    # Compute GTS Time Windows
    t_end = args.t_end if args.t_end is not None else max_observed_t
    t_split = args.t_split if args.t_split is not None else (t_end - (args.test_days * 86400))
    t_start = args.t_start if args.t_start is not None else (t_split - (args.train_days * 86400))

    print("=" * 80, flush=True)
    print(f"BUILDING GTS DATASET: {args.dataset_name.upper()}", flush=True)
    print(f"  Output Directory        : {output_dir}", flush=True)
    print(f"  Training Start (T_start): {t_start} ({datetime.fromtimestamp(t_start, tz=timezone.utc)})", flush=True)
    print(f"  Global Split   (T_split): {t_split} ({datetime.fromtimestamp(t_split, tz=timezone.utc)})", flush=True)
    print(f"  Reference End  (T_end)  : {t_end} ({datetime.fromtimestamp(t_end, tz=timezone.utc)})", flush=True)
    print(f"  Min User Core           : {args.min_user_core}", flush=True)
    print(f"  Min Warm Frequency      : {args.min_freq_warm}", flush=True)
    print("=" * 80, flush=True)

    # Filter interactions within [t_start, t_end]
    print(f"\n[Step 2] Filtering and deduplicating in window [{t_start}, {t_end}]...", flush=True)
    user_timeline = defaultdict(list)
    seen = set()
    for u, i, t in all_interactions:
        if t_start <= t <= t_end:
            key = (u, i, t)
            if key not in seen:
                seen.add(key)
                user_timeline[u].append((t, i))

    for u in user_timeline:
        user_timeline[u].sort(key=lambda x: x[0])

    total_window_inters = sum(len(h) for h in user_timeline.values())
    print(f"Retained {total_window_inters:,} unique interactions across {len(user_timeline):,} users.")

    if total_window_inters == 0:
        print("ERROR: Zero interactions fell within the window [t_start, t_end]! Check timestamps.")
        return

    # Build and export splits
    if is_amazon_mode and args.export_metadata:
        # Build dataset with Amazon specific metadata downloader
        from gts_core import run_iterative_k_core
        train_user_items = {u: [i for t, i in h if t_start <= t < t_split] for u, h in user_timeline.items() if len([i for t, i in h if t_start <= t < t_split]) >= args.min_user_core}
        filtered_train_users, warm_catalog = run_iterative_k_core(train_user_items, args.min_user_core, args.min_freq_warm)
        sorted_catalog = sorted(warm_catalog)
        item_map = {item: idx + 1 for idx, item in enumerate(sorted_catalog)}

        if not args.only_metadata:
            build_and_export_splits(
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
                output_dir=output_dir,
                delimiter=args.delimiter
            )

        cats_to_fetch = ALL_AMAZON_2023_CATEGORIES if not args.categories or "all" in args.categories else args.categories
        export_amazon_metadata(
            warm_catalog=warm_catalog,
            item_map=item_map,
            categories=cats_to_fetch,
            cache_dir=args.cache_dir,
            output_dir=output_dir,
            meta_threads=args.meta_threads,
            keep_meta_files=args.keep_meta_files
        )
    else:
        # Universal dataset build
        build_and_export_splits(
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
            output_dir=output_dir,
            metadata_path=args.metadata_path,
            meta_item_col=args.meta_item_col,
            meta_title_col=args.meta_title_col,
            meta_category_col=args.meta_category_col,
            delimiter=args.delimiter
        )


if __name__ == "__main__":
    main()
