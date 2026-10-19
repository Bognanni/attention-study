import argparse
import optuna
import os
import re
import subprocess
import sys


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
    reuse_item_embeddings={reuse_item_embeddings},
)
"""


def create_objective(args):
    def objective(trial):
        if args.model_type == 'gsasrec':
            # Smart Search Space for gSASRec:
            # Fix constants solidly validated by previous 25 trials
            sequence_length = 50
            num_blocks = 2
            train_batch_size = 128

            # Tune gSASRec specific & core parameters
            gbce_t = trial.suggest_categorical('gbce_t', [0.25, 0.5, 0.75, 1.0])
            negs_per_pos = trial.suggest_categorical('negs_per_pos', [64, 128, 256])
            dropout_rate = trial.suggest_categorical('dropout_rate', [0.3, 0.4, 0.5])
            embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128])
            num_heads = trial.suggest_categorical('num_heads', [1, 2])
            reuse_item_embeddings = trial.suggest_categorical('reuse_item_embeddings', [True, False])

        else:  # sasrec
            gbce_t = 0.0
            negs_per_pos = 1
            reuse_item_embeddings = False

            sequence_length = trial.suggest_categorical('sequence_length', [15, 30, 50])
            embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128])
            num_heads = trial.suggest_categorical('num_heads', [1, 2, 4])
            num_blocks = trial.suggest_int('num_blocks', 1, 2)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
            train_batch_size = trial.suggest_categorical('train_batch_size', [64, 128])

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
            reuse_item_embeddings=reuse_item_embeddings,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.patience
        )

        name_file_config = f"config_optuna_trial_{trial.number}.py"
        with open(name_file_config, "w") as f:
            f.write(generated_config)

        print(
            f"\n[Trial {trial.number}] Created {name_file_config} ({args.model_type.upper()}): "
            f"t={gbce_t}, negs={negs_per_pos}, emb={embedding_dim}, drop={dropout_rate}, "
            f"heads={num_heads}, tied_emb={reuse_item_embeddings}"
        )

        # Run training subprocess with explicit UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        command = [sys.executable, 'train_gsasrec.py', '--config', name_file_config]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', env=env)
        output_text = result.stdout
        error_text = result.stderr

        # Clean up temporary config file
        if os.path.exists(name_file_config):
            os.remove(name_file_config)

        # Parse Optuna score
        match = re.search(r'FINAL_SCORE_FOR_OPTUNA: ([0-9]+\.[0-9]+)', output_text)
        if match:
            score = float(match.group(1))
            print(f"[Trial {trial.number}] Success -> Validation Score (nDCG@10): {score:.6f}")
        else:
            # Fallback: search for metric in saved model filename
            metric_matches = re.findall(r'metric_([0-9]+\.[0-9]+)', output_text)
            if metric_matches:
                score = max(float(m) for m in metric_matches)
                print(f"[Trial {trial.number}] Parsed from checkpoint logs -> Score: {score:.6f}")
            else:
                score = 0.0
                print(f"[Trial {trial.number}] Score not found! Assigned 0.0")
                if error_text:
                    print("-" * 40)
                    print("Subprocess Error Output:")
                    print(error_text)
                    print("-" * 40)

        return score

    return objective


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization (HPO) for SASRec & gSASRec")
    parser.add_argument('--dataset', type=str, default='amazon_gts', help="Target dataset name")
    parser.add_argument('--model_type', type=str, default='gsasrec', choices=['sasrec', 'gsasrec'],
                        help="Model to tune: 'gsasrec' (G-BCE with multiple negatives) or 'sasrec'")
    parser.add_argument('--n_trials', type=int, default=15, help="Number of Optuna trials to run")
    parser.add_argument('--max_epochs', type=int, default=1000, help="Maximum epochs per trial (default: 1000)")
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

    print("=" * 70)
    print(f"STARTING HPO FOR: {args.dataset.upper()} | MODEL: {args.model_type.upper()}")
    print(f"Trials: {args.n_trials} | Max Epochs: {args.max_epochs} | Patience: {args.patience}")
    print(f"Storage: {args.storage} | Study: {args.study_name}")
    print("=" * 70)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction='maximize'
    )

    objective_fn = create_objective(args)
    study.optimize(objective_fn, n_trials=args.n_trials)

    print("\n" + "=" * 70)
    print(f"HPO COMPLETED FOR {args.dataset} ({args.model_type.upper()})!")
    print(f"Best Validation Score (nDCG@10): {study.best_value:.6f}")
    print("Best Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  - {param}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    main()