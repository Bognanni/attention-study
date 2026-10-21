import argparse
import optuna
import os
import re
import subprocess
import sys


# Configuration template passed dynamically to train_gsasrec.py
# Configured with R@50 (Recall@50 / HitRatio@50) and recommendation_limit=50
TEMPLATE_CONFIG = """from config import GSASRecExperimentConfig
from ir_measures import R, nDCG

config = GSASRecExperimentConfig(
    dataset_name='{dataset_name}',
    train_batch_size={train_batch_size}, 
    sequence_length={sequence_length}, 
    embedding_dim={embedding_dim},
    num_heads={num_heads},
    max_batches_per_epoch=100,
    max_epochs={max_epochs},
    early_stopping_patience={early_stopping_patience},
    num_blocks={num_blocks},
    dropout_rate={dropout_rate},
    negs_per_pos={negs_per_pos},
    gbce_t={gbce_t},
    reuse_item_embeddings=False,
    recommendation_limit=50,
    val_metric=R@50,
    metrics=[R@50, nDCG@50, nDCG@10, R@10, R@1],
)
"""

# Dynamically extract target validation metric from TEMPLATE_CONFIG to avoid hardcoded metric strings
val_metric_match = re.search(r'val_metric\s*=\s*([^,\n]+)', TEMPLATE_CONFIG)
TARGET_METRIC_LABEL = val_metric_match.group(1).strip() if val_metric_match else "Validation Metric"


class BestCheckpointManager:
    """
    Tracks and manages saved checkpoint files across trials.
    Retains only the single overall best model checkpoint on disk,
    deleting suboptimal checkpoints to conserve storage while keeping full metrics in Optuna.
    """
    def __init__(self, study=None):
        self.best_score = float('-inf')
        self.best_checkpoint_path = None
        if study is not None:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            if completed_trials:
                best_t = max(completed_trials, key=lambda t: t.value)
                self.best_score = best_t.value
                saved_cp = best_t.user_attrs.get('checkpoint', None)
                if saved_cp and os.path.exists(saved_cp):
                    self.best_checkpoint_path = saved_cp

    def update(self, trial, score, trial_checkpoint):
        if trial_checkpoint:
            trial.set_user_attr('checkpoint', trial_checkpoint)

        if score > self.best_score:
            print(f"[Checkpoint] New best score achieved: {score:.6f} (previous: {self.best_score:.6f})")
            # If a previous best checkpoint exists and differs, remove it
            if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path) and self.best_checkpoint_path != trial_checkpoint:
                try:
                    os.remove(self.best_checkpoint_path)
                    print(f"[Cleanup] Removed previous best checkpoint: {self.best_checkpoint_path}")
                except Exception as e:
                    print(f"[Cleanup] Warning: could not remove previous checkpoint {self.best_checkpoint_path}: {e}")
            self.best_score = score
            self.best_checkpoint_path = trial_checkpoint
            if trial_checkpoint:
                print(f"[Checkpoint] Retained new best model: {trial_checkpoint}")
        else:
            # Suboptimal trial: delete its checkpoint to free disk space
            if trial_checkpoint and os.path.exists(trial_checkpoint) and trial_checkpoint != self.best_checkpoint_path:
                try:
                    os.remove(trial_checkpoint)
                    print(f"[Cleanup] Removed suboptimal checkpoint: {trial_checkpoint}")
                except Exception as e:
                    print(f"[Cleanup] Warning: could not remove {trial_checkpoint}: {e}")

    def clean_suboptimal_checkpoint(self, output_text):
        checkpoint_matches = re.findall(r'Saving new best model to (.+\.pt)', output_text)
        if checkpoint_matches:
            cp = checkpoint_matches[-1].strip()
            if os.path.exists(cp) and cp != self.best_checkpoint_path:
                try:
                    os.remove(cp)
                    print(f"[Cleanup] Removed failed/pruned trial checkpoint: {cp}")
                except Exception:
                    pass


def print_leaderboard(study, top_n=15):
    """
    Print a sorted leaderboard of the best evaluated combinations.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed:
        print("\nNo completed trials to display.")
        return

    # Sort trials by value descending
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)

    print("\n" + "=" * 80)
    print(f"HYPERPARAMETER TUNING LEADERBOARD (Top {min(top_n, len(sorted_trials))} of {len(sorted_trials)} completed)")
    print("=" * 80)
    print(f"{'Rank':<4} | {'Trial':<6} | {'Val Score':<12} | Hyperparameters")
    print("-" * 80)

    for rank, trial in enumerate(sorted_trials[:top_n], start=1):
        params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
        val_str = f"{trial.value:.6f}"
        print(f"#{rank:<3} | #{trial.number:<5} | {val_str:<12} | {params_str}")

    print("=" * 80)


def create_objective(args, checkpoint_manager):
    def objective(trial):
        if args.model_type == 'gsasrec':
            # Constants for gSASRec on sparse datasets
            sequence_length = 50
            num_blocks = 2
            num_heads = 1
            train_batch_size = 128

            # Tune gSASRec specific & core parameters
            gbce_t = trial.suggest_categorical('gbce_t', [0.5, 0.75])
            negs_per_pos = trial.suggest_categorical('negs_per_pos', [128, 256])
            dropout_rate = trial.suggest_categorical('dropout_rate', [0.3, 0.5])
            embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128])

        else:  # sasrec
            gbce_t = 0.0
            negs_per_pos = 1

            # Fixed architectural constants based on empirical findings
            sequence_length = 50
            num_blocks = 2
            num_heads = 1
            train_batch_size = 128

            # Tune key parameters across a broad dropout scale
            embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128])
            dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.8, step=0.1)

        # Fill the configuration template
        generated_config = TEMPLATE_CONFIG.format(
            dataset_name=args.dataset,
            sequence_length=sequence_length,
            train_batch_size=train_batch_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            negs_per_pos=negs_per_pos,
            gbce_t=gbce_t,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.patience
        )

        name_file_config = f"config_optuna_trial_{trial.number}.py"
        with open(name_file_config, "w", encoding="utf-8") as f:
            f.write(generated_config)

        if args.model_type == 'gsasrec':
            print(
                f"\n[Trial {trial.number}] Created {name_file_config} (GSASREC): "
                f"t={gbce_t}, negs={negs_per_pos}, emb={embedding_dim}, drop={dropout_rate:.1f}"
            )
        else:
            print(
                f"\n[Trial {trial.number}] Created {name_file_config} (SASREC): "
                f"emb={embedding_dim}, drop={dropout_rate:.1f}, blocks={num_blocks}, "
                f"heads={num_heads}, batch={train_batch_size}"
            )

        # Run training subprocess with try...finally to ensure temporary config cleanup
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            command = [sys.executable, 'train_gsasrec.py', '--config', name_file_config]
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', env=env)
            output_text = result.stdout
            error_text = result.stderr
        finally:
            if os.path.exists(name_file_config):
                try:
                    os.remove(name_file_config)
                except Exception:
                    pass

        # Handle subprocess crash or failure
        if result.returncode != 0:
            print(f"[Trial {trial.number}] ERROR: Subprocess failed with exit code {result.returncode}")
            if error_text:
                print("-" * 40)
                print("Subprocess Error Output:")
                print(error_text.strip())
                print("-" * 40)
            checkpoint_manager.clean_suboptimal_checkpoint(output_text)
            raise optuna.TrialPruned(f"Subprocess failed with exit code {result.returncode}")

        # Parse validation score supporting standard float and scientific notation (e.g. 1.234e-05)
        match = re.search(r'FINAL_SCORE_FOR_OPTUNA:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)', output_text)
        if match:
            score = float(match.group(1))
            print(f"[Trial {trial.number}] Success -> Validation Score: {score:.6f}")
        else:
            # Fallback: search for metric in saved checkpoint logs
            metric_matches = re.findall(r'metric_([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\.pt', output_text)
            if metric_matches:
                score = max(float(m) for m in metric_matches)
                print(f"[Trial {trial.number}] Parsed from checkpoint logs -> Score: {score:.6f}")
            else:
                print(f"[Trial {trial.number}] Score not found in output! Pruning trial.")
                if error_text:
                    print("-" * 40)
                    print("Subprocess Error Output:")
                    print(error_text.strip())
                    print("-" * 40)
                checkpoint_manager.clean_suboptimal_checkpoint(output_text)
                raise optuna.TrialPruned("Validation metric score not found in training output")

        # Track and clean up checkpoints (keeping only the best model)
        checkpoint_matches = re.findall(r'Saving new best model to (.+\.pt)', output_text)
        trial_checkpoint = checkpoint_matches[-1].strip() if checkpoint_matches else None
        checkpoint_manager.update(trial, score, trial_checkpoint)

        return score

    return objective


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization (HPO) for SASRec & gSASRec")
    parser.add_argument('--dataset', type=str, default='amazon_gts', help="Target dataset name")
    parser.add_argument('--model_type', type=str, default='gsasrec', choices=['sasrec', 'gsasrec'],
                        help="Model to tune: 'gsasrec' (G-BCE with multiple negatives) or 'sasrec'")
    parser.add_argument('--n_trials', type=int, default=15, help="Number of Optuna trials to run")
    parser.add_argument('--max_epochs', type=int, default=10000, help="Maximum epochs per trial (default: 10000)")
    parser.add_argument('--patience', type=int, default=200, help="Early stopping patience in epochs (default: 200)")
    parser.add_argument('--storage', type=str, default=None, 
                        help="Optuna DB storage URL (default: sqlite:///hpo_<dataset>_<model_type>.db)")
    parser.add_argument('--study_name', type=str, default=None, 
                        help="Optuna study name (default: hpo_<dataset>_<model_type>)")
    args = parser.parse_args()

    if args.storage is None:
        args.storage = f"sqlite:///hpo_{args.dataset}_{args.model_type}.db"
    if args.study_name is None:
        args.study_name = f"hpo_{args.dataset}_{args.model_type}"

    print("=" * 80)
    print(f"STARTING HPO FOR: {args.dataset.upper()} | MODEL: {args.model_type.upper()}")
    print(f"Validation Metric:       {TARGET_METRIC_LABEL}")
    print(f"Trials: {args.n_trials} | Max Epochs: {args.max_epochs} | Patience: {args.patience}")
    print(f"Storage: {args.storage} | Study: {args.study_name}")
    print("=" * 80)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction='maximize'
    )

    checkpoint_manager = BestCheckpointManager(study)
    objective_fn = create_objective(args, checkpoint_manager)

    try:
        study.optimize(objective_fn, n_trials=args.n_trials, catch=(Exception,))
    except KeyboardInterrupt:
        print("\n\n" + "!" * 80)
        print("MANUAL INTERRUPTION DETECTED (Ctrl+C)!")
        print("All completed trials have been safely saved to SQLite storage.")
        print("!" * 80)

    print("\n" + "=" * 80)
    print(f"HPO SUMMARY FOR {args.dataset.upper()} ({args.model_type.upper()})")
    print("=" * 80)

    # Display ranked leaderboard of best models
    print_leaderboard(study, top_n=15)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        best_trial = study.best_trial
        print(f"\nBEST OVERALL RESULT:")
        print(f"  * Best Trial:            #{best_trial.number}")
        print(f"  * Best Validation Score:  {best_trial.value:.6f}")
        print(f"  * Preserved Checkpoint:   {checkpoint_manager.best_checkpoint_path}")
        print("  * Best Hyperparameters:")
        for param, value in best_trial.params.items():
            print(f"    - {param}: {value}")
    else:
        print("No completed trials found in this study yet.")

    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nParameter Importance (fANOVA):")
        for param, importance in importances.items():
            print(f"  * {param:22s}: {importance * 100:.1f}%")
    except Exception as e:
        print(f"\nParameter Importance could not be computed (needs >= 2 completed trials): {e}")

    print("=" * 80)


if __name__ == "__main__":
    main()
