"""
Inspect and Decode Sequence Interactions with Product Metadata
==============================================================
This script visualizes user purchase sequences from the Amazon GTS dataset
(train, val, or test splits) and decodes item IDs into human-readable product
titles, categories, brands, prices, and ratings using the metadata catalog.

Usage Examples:
  # Show the first 5 test sequences:
  python inspect_sequences.py --split test --n 5

  # Show a specific sequence by 1-indexed line number:
  python inspect_sequences.py --split test --line 42

  # Show multiple specific lines with detailed product metadata:
  python inspect_sequences.py --split val --lines 1 10 50 --detailed

  # Inspect 3 random training sequences:
  python inspect_sequences.py --split train --random 3
"""

import os
import sys
import csv
import random
import argparse
from pathlib import Path
from collections import Counter


def load_metadata(metadata_path: Path) -> dict:
    """
    Loads item metadata from item_metadata.csv into a dictionary keyed by integer item_id.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at: {metadata_path}\n"
            "Please ensure 'item_metadata.csv' is present in 'amazon_gts_metadata/' "
            "or 'datasets/amazon_gts/'."
        )

    print(f"Loading metadata from {metadata_path} ...", end=" ", flush=True)
    metadata = {}
    with open(metadata_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                item_id = int(row["item_id"])
                metadata[item_id] = row
            except (ValueError, KeyError):
                continue
    print(f"Done! ({len(metadata):,} items loaded)", flush=True)
    return metadata


def format_item_line(pos_label: str, item_id: int, meta: dict, max_title_len: int = 90, detailed: bool = False) -> str:
    """
    Formats a single item into a human-readable string.
    """
    info = meta.get(item_id, {})
    title = (info.get("title") or f"Item {item_id}").strip()
    if max_title_len > 0 and len(title) > max_title_len:
        title = title[: max_title_len - 3] + "..."

    category = info.get("main_category") or "Unknown"
    asin = info.get("parent_asin") or "N/A"

    if not detailed:
        return f"{pos_label} [ID: {item_id:6d}] [{category}] {title}"

    # Detailed formatting
    store = info.get("store") or "N/A"
    avg_rating = info.get("average_rating") or "-"
    ratings_count = info.get("rating_number") or "-"
    price = info.get("price") or "-"
    price_str = f"${price}" if price and price != "-" else "N/A"

    line = (
        f"{pos_label} [ID: {item_id:6d} | ASIN: {asin}]\n"
        f"      Category : {category}\n"
        f"      Title    : {title}\n"
        f"      Brand/By : {store} | Rating: {avg_rating}* ({ratings_count} reviews) | Price: {price_str}"
    )
    return line


def display_sequence(line_number: int, input_ids: list, target_id: int or None, metadata: dict, max_title_len: int = 90, detailed: bool = False):
    """
    Prints a formatted visualization of a single user interaction history and its target.
    """
    print("\n" + "=" * 90)
    target_str = f"Target ID: {target_id}" if target_id is not None else "No explicit target (Train)"
    print(f"  SEQUENCE #{line_number:,} (Length: {len(input_ids)} items | {target_str})")
    print("=" * 90)

    # Category overview
    categories = [metadata.get(i, {}).get("main_category", "Unknown") for i in input_ids]
    cat_counts = Counter(categories)
    cat_summary = ", ".join(f"{cat} ({count})" for cat, count in cat_counts.most_common(4))
    print(f"Categories in History: {cat_summary}")

    if target_id is not None:
        target_cat = metadata.get(target_id, {}).get("main_category", "Unknown")
        is_repurchase = target_id in set(input_ids)
        repurchase_tag = " [!] REPURCHASE (already in history)" if is_repurchase else " [NEW Item]"
        same_cat = "Same Category" if target_cat in cat_counts else "Cross-Category"
        print(f"Target Category      : {target_cat} ({same_cat}){repurchase_tag}")

    print("-" * 90)
    print("USER PURCHASE HISTORY (Chronological Order):")

    seq_len = len(input_ids)
    for idx, item_id in enumerate(input_ids, start=1):
        rel_pos = f"t-{seq_len - idx}" if idx < seq_len else "t-0 (Latest)"
        pos_label = f"  {idx:2d}. ({rel_pos:11s})"
        print(format_item_line(pos_label, item_id, metadata, max_title_len, detailed))

    if target_id is not None:
        print("\nGROUND TRUTH TARGET (Item to Predict Next):")
        pos_label = "  >>> (Next-Item Target)"
        print(format_item_line(pos_label, target_id, metadata, max_title_len, detailed))

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and decode sequence interactions with product titles and metadata."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to inspect: 'test' (default), 'val', or 'train'."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="datasets/amazon_gts",
        help="Path to the dataset directory containing train/val/test splits."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="amazon_gts_metadata/item_metadata.csv",
        help="Path to the item_metadata.csv file."
    )
    parser.add_argument(
        "-n", "--first",
        type=int,
        default=None,
        help="Display the first N sequences."
    )
    parser.add_argument(
        "-l", "--line",
        type=int,
        default=None,
        help="Display a specific 1-indexed sequence line number (e.g., --line 10)."
    )
    parser.add_argument(
        "--lines",
        type=int,
        nargs="+",
        default=None,
        help="Display multiple specific 1-indexed sequence line numbers (e.g., --lines 1 5 10)."
    )
    parser.add_argument(
        "--random",
        type=int,
        default=None,
        help="Display K randomly chosen sequences."
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Display extended product details (ASIN, store/brand, price, ratings)."
    )
    parser.add_argument(
        "--max_title_len",
        type=int,
        default=85,
        help="Maximum characters to show for item titles (default: 85, set 0 for full)."
    )
    args = parser.parse_args()

    # Locate metadata path
    meta_path = Path(args.metadata_path)
    if not meta_path.exists():
        fallback_path = Path("datasets/amazon_gts/item_metadata.csv")
        if fallback_path.exists():
            meta_path = fallback_path

    metadata = load_metadata(meta_path)

    # Locate split files
    split_dir = Path(args.dataset_dir) / args.split
    input_file = split_dir / "input.txt"
    output_file = split_dir / "output.txt"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    has_output = output_file.exists()

    # Determine which line indices (1-indexed) to inspect
    target_lines = set()
    if args.line is not None:
        target_lines.add(args.line)
    if args.lines is not None:
        target_lines.update(args.lines)

    # Count total available lines in input file
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    if args.random is not None and args.random > 0:
        sample_k = min(args.random, total_lines)
        random.seed(42)
        target_lines.update(random.sample(range(1, total_lines + 1), sample_k))
    elif not target_lines:
        n_show = args.first if args.first is not None else 5
        target_lines.update(range(1, min(n_show, total_lines) + 1))

    sorted_target_lines = sorted(l for l in target_lines if 1 <= l <= total_lines)
    if not sorted_target_lines:
        print(f"Error: Selected line number(s) out of range (Total lines in {args.split}: {total_lines:,}).")
        sys.exit(1)

    print(
        f"\nInspecting {len(sorted_target_lines)} sequence(s) from '{args.split}' split "
        f"(Total sequences available: {total_lines:,}) ...\n"
    )

    # Stream through files and display targeted lines
    f_in = open(input_file, "r", encoding="utf-8")
    f_out = open(output_file, "r", encoding="utf-8") if has_output else None

    try:
        current_line = 0
        target_set = set(sorted_target_lines)
        max_needed_line = max(sorted_target_lines)

        while current_line < max_needed_line:
            current_line += 1
            line_in = f_in.readline()
            if not line_in:
                break
            line_out = f_out.readline() if f_out else None

            if current_line in target_set:
                input_ids = [int(x) for x in line_in.strip().split()]
                target_id = int(line_out.strip()) if line_out and line_out.strip() else None
                display_sequence(
                    line_number=current_line,
                    input_ids=input_ids,
                    target_id=target_id,
                    metadata=metadata,
                    max_title_len=args.max_title_len,
                    detailed=args.detailed
                )
    finally:
        f_in.close()
        if f_out:
            f_out.close()


if __name__ == "__main__":
    main()
