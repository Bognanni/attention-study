import math
import os
import re
import subprocess
import sys
import optuna

# ==============================================================================
# CONFIGURATION TEMPLATE & DYNAMIC METRIC EXTRACTION
# ==============================================================================
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
    reuse_item_embeddings={reuse_item_embeddings},
    recommendation_limit=50,
    val_metric=R@50,
    metrics=[R@50, nDCG@50, nDCG@10, R@10, R@1],
)
"""


def get_target_metric_label(template_config: str = TEMPLATE_CONFIG) -> str:
    """
    Dynamically extracts the target validation metric label from the configuration template.
    Avoids hardcoding metric strings across logs and print statements.
    """
    match = re.search(r'val_metric\s*=\s*([^,\n]+)', template_config)
    return match.group(1).strip() if match else "Validation Metric"


TARGET_METRIC_LABEL = get_target_metric_label(TEMPLATE_CONFIG)


# ==============================================================================
# UNIFIED SEARCH SPACE DEFINITION
# ==============================================================================
# Shared base transformer architectural configuration
COMMON_ARCH_GRID = {
    'embedding_dim': [64, 128, 256, 512],
    'num_blocks': [2, 3, 4],
    'num_heads': [1, 2, 4],
    'sequence_length': [50],
    'train_batch_size': [128],
    'reuse_item_embeddings': [False, True],
}


def get_search_grid(model_type: str) -> dict:
    """
    Returns the search space grid for the specified model type, building on top
    of the common transformer architecture parameters.
    - sasrec:  fixed gbce_t=0.0 and negs_per_pos=1, explores broad dropout scale
    - gsasrec: explores calibrated temperatures [0.5, 0.75], multiple negatives [128, 256]
    """
    grid = dict(COMMON_ARCH_GRID)
    model_key = model_type.strip().lower()

    if model_key == 'sasrec':
        grid.update({
            'dropout_rate': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'gbce_t': [0.0],
            'negs_per_pos': [1],
        })
    elif model_key == 'gsasrec':
        grid.update({
            'dropout_rate': [0.3, 0.6, 0.8],
            'gbce_t': [0.5, 0.75],
            'negs_per_pos': [128, 256],
        })
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Supported values are 'sasrec' and 'gsasrec'.")

    return grid


def calculate_total_combinations(grid: dict) -> int:
    """
    Computes the total number of combinations in the Cartesian product of the grid.
    """
    return math.prod(len(v) for v in grid.values())


# ==============================================================================
# BEST CHECKPOINT MANAGER
# ==============================================================================
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

    def update(self, trial, score: float, trial_checkpoint: str):
        """
        Updates the best score and manages checkpoint files.
        - If the current trial's score exceeds the best score, retain its checkpoint and delete the previous best (if different).
        - If the current trial is suboptimal, delete its checkpoint to free disk space.
        """
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

    def clean_suboptimal_checkpoint(self, output_text: str):
        """
        Parses output text for checkpoints that might have been saved before a trial failure/pruning
        and removes them if they are not the best model.
        """
        checkpoint_matches = re.findall(r'Saving new best model to (.+\.pt)', output_text)
        if checkpoint_matches:
            cp = checkpoint_matches[-1].strip()
            if os.path.exists(cp) and cp != self.best_checkpoint_path:
                try:
                    os.remove(cp)
                    print(f"[Cleanup] Removed failed/pruned trial checkpoint: {cp}")
                except Exception:
                    pass


# ==============================================================================
# SUBPROCESS EXECUTION & SCORE PARSING
# ==============================================================================
def run_training_subprocess(trial, config_content: str, checkpoint_manager: BestCheckpointManager, config_prefix: str = "trial", verbose: bool = False) -> float:
    """
    Executes a single training trial in an isolated subprocess.
    - Writes a temporary configuration file with guaranteed cleanup in a finally block
    - Runs train_gsasrec.py with UTF-8 encoding
    - Parses final validation score and handles subprocess crashes/pruning
    - Updates BestCheckpointManager to retain only the top model
    """
    name_file_config = f"config_{config_prefix}_{trial.number}.py"
    with open(name_file_config, "w", encoding="utf-8") as f:
        f.write(config_content)

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    command = [sys.executable, 'train_gsasrec.py', '--config', name_file_config]

    output_text = ""
    error_text = ""
    return_code = 0

    try:
        if verbose:
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

    # Handle subprocess failure
    if return_code != 0:
        print(f"[Trial {trial.number}] ERROR: Subprocess failed with exit code {return_code}")
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


# ==============================================================================
# REPORTING & EXPORT UTILITIES
# ==============================================================================
def print_leaderboard(study, top_n: int = 15, title: str = "LEADERBOARD"):
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
    print(f"{title} (Top {min(top_n, len(sorted_trials))} of {len(sorted_trials)} completed)")
    print("=" * 80)
    print(f"{'Rank':<4} | {'Trial':<6} | {'Val Score':<12} | Hyperparameters")
    print("-" * 80)

    for rank, trial in enumerate(sorted_trials[:top_n], start=1):
        params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
        val_str = f"{trial.value:.6f}"
        print(f"#{rank:<3} | #{trial.number:<5} | {val_str:<12} | {params_str}")

    print("=" * 80)


def export_to_csv(study, filepath: str):
    """
    Export all trials to a CSV file.
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


def print_parameter_importances(study):
    """
    Computes and displays fANOVA parameter importances for completed trials.
    """
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nParameter Importance (fANOVA):")
        for param, importance in importances.items():
            print(f"  * {param:22s}: {importance * 100:.1f}%")
    except Exception as e:
        print(f"\nParameter Importance could not be computed (needs >= 2 completed trials): {e}")
