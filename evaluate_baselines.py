import argparse
import os
import random
import torch
from collections import Counter
from typing import List, Set, Dict, Any

from dataset_utils import get_num_items, get_test_dataloader
from eval_utils import evaluate
from utils import get_device, load_config
from config import GSASRecExperimentConfig


class MostPopRecommender:
    """
    Naive baseline that always recommends the globally most frequent items
    observed in the training set, filtering out items the user has already interacted with.
    """
    def __init__(self, popular_items: List[int]):
        self.popular_items = popular_items

    def eval(self):
        """
        No training required for MostPopRecommender; this method is a placeholder for API consistency
        """
        pass

    def get_predictions(self, data: torch.Tensor, limit: int, rated: List[List[int]] = None):
        """
        Generates recommendations for each user in the batch based on global item popularity.
        Ensures that already interacted items are not recommended.
        """
        bs = data.size(0)
        batch_items = []
        batch_scores = []

        for b in range(bs):
            user_rated: Set[int] = set(rated[b]) if rated is not None else set()
            recs = []
            for item in self.popular_items:
                if item not in user_rated:
                    recs.append(item)
                    if len(recs) == limit:
                        break
            # Fallback if catalog/interactions are somehow smaller than limit
            while len(recs) < limit:
                recs.append(0)

            batch_items.append(recs)
            # Monotonically decreasing scores for ranking
            batch_scores.append(list(range(limit, 0, -1)))

        return (
            torch.tensor(batch_items, dtype=torch.long, device=data.device),
            torch.tensor(batch_scores, dtype=torch.float32, device=data.device)
        )


class RandomRecommender:
    """
    Naive baseline that recommends uniformly random items from the catalog,
    filtering out items the user has already interacted with.
    """
    def __init__(self, num_items: int, seed: int = 42):
        self.num_items = num_items
        self.rng = random.Random(seed)

    def eval(self):
        """
            No training required for MostPopRecommender; this method is a placeholder for API consistency
        """
        pass

    def get_predictions(self, data: torch.Tensor, limit: int, rated: List[List[int]] = None):
        """
        Generates random recommendations for each user in the batch, ensuring that
        already interacted items are not recommended.
        """
        bs = data.size(0)
        batch_items = []
        batch_scores = []

        for b in range(bs):
            user_rated: Set[int] = set(rated[b]) if rated is not None else set()
            recs = []
            # Sample until we have 'limit' unique items not in rated
            attempts = 0
            max_attempts = limit * 100
            while len(recs) < limit and attempts < max_attempts:
                sampled = self.rng.randint(1, self.num_items)
                if sampled not in user_rated and sampled not in recs:
                    recs.append(sampled)
                attempts += 1

            # Fallback if catalog is saturated
            while len(recs) < limit:
                recs.append(self.rng.randint(1, self.num_items))

            batch_items.append(recs)
            # Decreasing scores
            batch_scores.append(list(range(limit, 0, -1)))

        return (
            torch.tensor(batch_items, dtype=torch.long, device=data.device),
            torch.tensor(batch_scores, dtype=torch.float32, device=data.device)
        )


def compute_item_popularity(train_file_path: str) -> List[int]:
    """
    Counts item occurrences in train/input.txt and returns item IDs sorted by descending popularity.
    """
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"Train file not found at: {train_file_path}")

    counts = Counter()
    with open(train_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            counts.update(int(t) for t in tokens)

    # Sort descending by count, tie-break by item_id
    sorted_items = [item for item, _ in counts.most_common()]
    return sorted_items


def main():
    parser = argparse.ArgumentParser(description="Evaluate Naive Baselines (Random & MostPop)")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Dataset name (e.g. amazon_gts, ml1m, beauty, video_games)")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to experiment config file (e.g. config_amazon_gts.py)")
    parser.add_argument('--baselines', type=str, nargs='+', default=['random', 'mostpop'],
                        choices=['random', 'mostpop'], help="Baselines to evaluate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Determine configuration
    if args.config is not None:
        config = load_config(args.config)
    elif args.dataset is not None:
        candidate_config = f"config_{args.dataset}.py"
        if os.path.exists(candidate_config):
            config = load_config(candidate_config)
        else:
            config = GSASRecExperimentConfig(dataset_name=args.dataset)
    else:
        # Default fallback
        args.dataset = "amazon_gts"
        config = load_config("config_amazon_gts.py")

    dataset_name = config.dataset_name
    device = get_device()
    num_items = get_num_items(dataset_name)

    print("=" * 80)
    print(f"EVALUATING NAIVE BASELINES ON DATASET: {dataset_name}")
    print(f"Catalog Items: {num_items:,} | Device: {device} | Recommendation Limit: {config.recommendation_limit}")
    print("=" * 80)

    # Load test dataloader
    test_dataloader = get_test_dataloader(dataset_name, batch_size=config.eval_batch_size, max_length=config.sequence_length)

    models_to_eval = {}

    if 'random' in args.baselines:
        random_model = RandomRecommender(num_items=num_items, seed=args.seed)
        models_to_eval["Random Recommender"] = random_model

    if 'mostpop' in args.baselines:
        train_file = os.path.join("datasets", dataset_name, "train", "input.txt")
        print(f"Computing item popularity from: {train_file} ...")
        popular_items = compute_item_popularity(train_file)
        print(f"Found {len(popular_items):,} unique interacting items in train set.")
        mostpop_model = MostPopRecommender(popular_items=popular_items)
        models_to_eval["MostPop Recommender"] = mostpop_model

    # Evaluation loop
    results: Dict[str, Dict[str, float]] = {}

    for model_name, model in models_to_eval.items():
        print(f"\nRunning evaluation for: {model_name}...")
        raw_res = evaluate(
            model=model,
            data_loader=test_dataloader,
            metrics=config.metrics,
            limit=config.recommendation_limit,
            filter_rated=config.filter_rated,
            device=device
        )
        # Format metric keys as readable strings
        formatted_res = {str(k): float(v) for k, v in raw_res.items()}
        results[model_name] = formatted_res

    # Display final results table
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION RESULTS ({dataset_name})")
    print("=" * 80)

    # Collect all metric names
    all_metrics = list(list(results.values())[0].keys()) if results else []

    header = f"{'Model':<25}" + "".join(f"{m:<18}" for m in all_metrics)
    print(header)
    print("-" * len(header))

    for model_name, metric_dict in results.items():
        row = f"{model_name:<25}" + "".join(f"{metric_dict.get(m, 0.0):<18.6f}" for m in all_metrics)
        print(row)

    print("=" * 80)
    print("Raw output dictionary:")
    print(results)


if __name__ == '__main__':
    main()
