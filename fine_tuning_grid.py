import argparse
import math
import os
import re
import subprocess
import sys
import optuna
from optuna.samplers import GridSampler


# ==============================================================================
# HYPERPARAMETER GRID DEFINITIONS (EXHAUSTIVE SEARCH SPACE)
# ==============================================================================
# The Cartesian product across all parameters is tested systematically.
# Fixed architectural constants and defaults are kept consistent with config.py.
# ==============================================================================

DEFAULT_GRID_SASREC = {
    'dropout_rate': [0.3, 0.5, 0.7],
    'embedding_dim': [64, 128],
    'num_blocks': [1, 2],
    'num_heads': [1, 2],
    'sequence_length': [50],
    'train_batch_size': [128],
}

DEFAULT_GRID_GSASREC = {
    'gbce_t': [0.25, 0.75],
    'negs_per_pos': [64, 256],
    'dropout_rate': [0.3, 0.5],
    'embedding_dim': [64, 128],
    'num_heads': [1],
}

# Configuration template passed dynamically to train_gsasrec.py
# Uses default validation metric (nDCG@10), cutoff, and metric list defined in config.py
TEMPLATE_CONFIG = """from config import GSASRecExperimentConfig

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
)
"""


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


def calculate_total_combinations(grid):
    """
    Compute the total number of combinations in the Cartesian product.
    """
    return math.prod(len(v) for v in grid.values())


def create_objective(args, grid, total_combinations, checkpoint_manager):
    """
    Build the objective function for Exhaustive Grid Search.
    """

    # Fixed parameters outside the respective search grid
    if args.model_type == 'sasrec':
        fixed_params = {
            'gbce_t': 0.0,
            'negs_per_pos': 1,
        }
    else:  # gsasrec
        fixed_params = {
            'sequence_length': 50,
            'num_blocks': 2,
            'train_batch_size': 128,
        }

    def objective(trial):
        # Sample the exact combination assigned by GridSampler
        sampled_params = {}
        for param_name, values_list in grid.items():
            sampled_params[param_name] = trial.suggest_categorical(param_name, values_list)

        # Merge sampled and fixed parameters
        all_params = {**fixed_params, **sampled_params}

        # Generate temporary configuration file
        generated_config = TEMPLATE_CONFIG.format(
            dataset_name=args.dataset,
            sequence_length=all_params['sequence_length'],
            train_batch_size=all_params['train_batch_size'],
            embedding_dim=all_params['embedding_dim'],
            num_heads=all_params['num_heads'],
            num_blocks=all_params['num_blocks'],
            dropout_rate=all_params['dropout_rate'],
            negs_per_pos=all_params['negs_per_pos'],
            gbce_t=all_params['gbce_t'],
            max_epochs=args.max_epochs,
            early_stopping_patience=args.patience,
        )

        name_file_config = f"config_grid_trial_{trial.number}.py"
        with open(name_file_config, "w", encoding="utf-8") as f:
            f.write(generated_config)

        trial_idx = trial.number + 1
        print("\n" + "-" * 80)
        print(f"[Grid Trial {trial_idx}/{total_combinations}] Testing configuration ({args.model_type.upper()}):")
        for k, v in sorted(sampled_params.items()):
            print(f"  * {k:22s}: {v}")
        print("-" * 80)

        # Execute training subprocess with guaranteed temporary file cleanup
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        command = [sys.executable, 'train_gsasrec.py', '--config', name_file_config]

        try:
            if args.verbose:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', env=env
                )
                output_lines = []
                for line in process.stdout:
                    print(line, end='', flush=True)
                    output_lines.append(line)
                process.wait()
                return_code = process.returncode
                output_text = "".join(output_lines)
                error_text = ""
            else:
                result = subprocess.run(
                    command, capture_output=True, text=True, encoding='utf-8', env=env
                )
                return_code = result.returncode
                output_text = result.stdout
                error_text = result.stderr
        finally:
            if os.path.exists(name_file_config):
                try:
                    os.remove(name_file_config)
                except Exception:
                    pass

        # Handle subprocess crash or failure
        if return_code != 0:
            print(f"[Grid Trial {trial_idx}] ERROR: Subprocess failed with exit code {return_code}")
            if error_text:
                print("-" * 40)
                print("Subprocess Error Output:")
                print(error_text.strip())
                print("-" * 40)
            checkpoint_manager.clean_suboptimal_checkpoint(output_text)
            raise optuna.TrialPruned(f"Subprocess failed with exit code {return_code}")

        # Parse validation score supporting standard float and scientific notation (e.g. 1.234e-05)
        match = re.search(r'FINAL_SCORE_FOR_OPTUNA:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)', output_text)
        if match:
            score = float(match.group(1))
            print(f"[Grid Trial {trial_idx}] Success -> Validation Score (nDCG@10): {score:.6f}")
        else:
            # Fallback: search for metric in saved checkpoint logs
            metric_matches = re.findall(r'metric_([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\.pt', output_text)
            if metric_matches:
                score = max(float(m) for m in metric_matches)
                print(f"[Grid Trial {trial_idx}] Parsed from checkpoint logs -> Score: {score:.6f}")
            else:
                print(f"[Grid Trial {trial_idx}] WARNING: No score found in output! Pruning trial.")
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
    print(f"GRID SEARCH LEADERBOARD (Top {min(top_n, len(sorted_trials))} of {len(sorted_trials)} completed)")
    print("=" * 80)
    print(f"{'Rank':<4} | {'Trial':<6} | {'Score (nDCG@10)':<16} | Hyperparameters")
    print("-" * 80)

    for rank, trial in enumerate(sorted_trials[:top_n], start=1):
        params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
        val_str = f"{trial.value:.6f}"
        print(f"#{rank:<3} | #{trial.number:<5} | {val_str:<16} | {params_str}")

    print("=" * 80)


def export_to_csv(study, filepath):
    """
    Export all Grid Search trials to a CSV file.
    """
    try:
        df = study.trials_dataframe()
        if not df.empty:
            # Clean up column names for readability
            col_map = {c: c.replace('params_', '') for c in df.columns if c.startswith('params_')}
            df.rename(columns=col_map, inplace=True)
            if 'value' in df.columns:
                df.sort_values(by='value', ascending=False, inplace=True)
            df.to_csv(filepath, index=False)
            print(f"\nFull results exported to CSV: {filepath}")
    except Exception as e:
        print(f"\nFailed to export CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive Grid Search for SASRec and gSASRec (Full Cartesian Product)"
    )
    parser.add_argument('--dataset', type=str, default='amazon_gts',
                        help="Target dataset name (default: amazon_gts)")
    parser.add_argument('--model_type', type=str, default='sasrec', choices=['sasrec', 'gsasrec'],
                        help="Model type: 'sasrec' or 'gsasrec' (default: sasrec)")
    parser.add_argument('--max_epochs', type=int, default=10000,
                        help="Maximum epochs per trial (default: 10000)")
    parser.add_argument('--patience', type=int, default=200,
                        help="Early stopping patience in epochs (default: 200)")
    parser.add_argument('--n_trials', type=int, default=None,
                        help="Number of combinations to execute in this session (default: all remaining)")
    parser.add_argument('--storage', type=str, default=None,
                        help="Optuna SQLite storage URL (default: sqlite:///grid_<dataset>_<model_type>.db)")
    parser.add_argument('--study_name', type=str, default=None,
                        help="Optuna study name (default: grid_<dataset>_<model_type>)")
    parser.add_argument('--csv_output', type=str, default=None,
                        help="Path for exported CSV results (default: grid_results_<dataset>_<model_type>.csv)")
    parser.add_argument('--verbose', action='store_true',
                        help="Stream epoch-by-epoch training logs live to stdout")

    args = parser.parse_args()

    # Select search grid
    if args.model_type == 'sasrec':
        grid = DEFAULT_GRID_SASREC
    else:
        grid = DEFAULT_GRID_GSASREC

    total_combinations = calculate_total_combinations(grid)

    # Default storage, study, and CSV file names
    if args.storage is None:
        args.storage = f"sqlite:///grid_{args.dataset}_{args.model_type}.db"
    if args.study_name is None:
        args.study_name = f"grid_{args.dataset}_{args.model_type}"
    if args.csv_output is None:
        args.csv_output = f"grid_results_{args.dataset}_{args.model_type}.csv"

    print("=" * 80)
    print(f"EXHAUSTIVE GRID SEARCH: {args.dataset.upper()} | MODEL: {args.model_type.upper()}")
    print("=" * 80)
    print("Grid Parameters:")
    for k, v in grid.items():
        print(f"  * {k:22s} ({len(v)} values): {v}")
    print("-" * 80)
    print(f"Total Grid Combinations:   {total_combinations}")
    print("Validation Metric:         nDCG@10 (default from config.py)")
    print(f"Max Epochs per Trial:     {args.max_epochs}")
    print(f"Early Stopping Patience:  {args.patience}")
    print(f"Persistent Database:      {args.storage}")
    print(f"Study Name:               {args.study_name}")
    print(f"CSV Output:               {args.csv_output}")
    print("=" * 80)

    # Initialize Optuna GridSampler
    sampler = GridSampler(grid)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        load_if_exists=True,
        direction='maximize'
    )

    checkpoint_manager = BestCheckpointManager(study)

    # Check progress status
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    num_completed = len(completed_trials)

    if sampler.is_exhausted(study):
        print(f"\nALL {total_combinations} grid combinations have already been completed in this database!")
        print_leaderboard(study, top_n=15)
        export_to_csv(study, args.csv_output)
        return

    unvisited_ids = sampler._get_unvisited_grid_ids(study)
    remaining_count = len(unvisited_ids)
    print(f"\nStudy Progress:")
    print(f"  * Already completed:  {num_completed}/{total_combinations}")
    print(f"  * Remaining:         {remaining_count}/{total_combinations}")

    # Determine number of trials to run in this session
    if args.n_trials is not None:
        trials_to_run = min(args.n_trials, remaining_count)
        print(f"  * Scheduled to run in this session: {trials_to_run}")
    else:
        trials_to_run = remaining_count
        print(f"  * Scheduled to run: ALL remaining combinations ({trials_to_run})")

    objective_fn = create_objective(args, grid, total_combinations, checkpoint_manager)

    try:
        study.optimize(objective_fn, n_trials=trials_to_run, catch=(Exception,))
    except KeyboardInterrupt:
        print("\n\n" + "!" * 80)
        print("MANUAL INTERRUPTION DETECTED (Ctrl+C)!")
        print("All completed trials have been safely saved to SQLite storage.")
        print("You can re-run the script at any time to resume from the exact stopping point.")
        print("!" * 80)

    # Final summary and leaderboard
    print("\n" + "=" * 80)
    print(f"GRID SEARCH SUMMARY FOR {args.dataset.upper()} ({args.model_type.upper()})")
    print("=" * 80)

    # Display ranked leaderboard of best models
    print_leaderboard(study, top_n=15)
    export_to_csv(study, args.csv_output)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        best_trial = study.best_trial
        print(f"\nBEST OVERALL RESULT:")
        print(f"  * Best Trial:            #{best_trial.number}")
        print(f"  * Validation Score:       {best_trial.value:.6f}")
        print(f"  * Preserved Checkpoint:   {checkpoint_manager.best_checkpoint_path}")
        print("  * Best Hyperparameters:")
        for param, val in best_trial.params.items():
            print(f"    - {param}: {val}")

        # Compute parameter importances (fANOVA)
        try:
            importances = optuna.importance.get_param_importances(study)
            print("\nParameter Importance (fANOVA):")
            for param, importance in importances.items():
                print(f"  * {param:22s}: {importance * 100:.1f}%")
        except Exception as e:
            print(f"\nParameter importance could not be computed: {e}")

    print("=" * 80)


if __name__ == "__main__":
    main()
