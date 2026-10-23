import argparse
import optuna

from hpo_utils import (
    TEMPLATE_CONFIG,
    TARGET_METRIC_LABEL,
    get_search_grid,
    BestCheckpointManager,
    print_leaderboard,
    export_to_csv,
    print_parameter_importances,
    run_training_subprocess,
)


def create_objective(args, checkpoint_manager: BestCheckpointManager):
    """
    Creates the objective function for Optuna hyperparameter optimization.
    Samples variable parameters from get_search_grid(args.model_type) and fixes
    single-value parameters automatically.
    """
    grid = get_search_grid(args.model_type)

    def objective(trial):
        sampled_params = {}
        for param_name, values in grid.items():
            if len(values) > 1:
                # Variable hyperparameter: let Optuna sampler (TPE/Random) suggest
                sampled_params[param_name] = trial.suggest_categorical(param_name, values)
            else:
                # Fixed constant in the grid: use the single value directly
                sampled_params[param_name] = values[0]

        # Fill the configuration template
        generated_config = TEMPLATE_CONFIG.format(
            dataset_name=args.dataset,
            sequence_length=sampled_params['sequence_length'],
            train_batch_size=sampled_params['train_batch_size'],
            embedding_dim=sampled_params['embedding_dim'],
            num_heads=sampled_params['num_heads'],
            num_blocks=sampled_params['num_blocks'],
            dropout_rate=sampled_params['dropout_rate'],
            negs_per_pos=sampled_params['negs_per_pos'],
            gbce_t=sampled_params['gbce_t'],
            max_epochs=args.max_epochs,
            early_stopping_patience=args.patience,
            reuse_item_embeddings=sampled_params['reuse_item_embeddings'],
        )

        trial_label = args.model_type.upper()
        print(f"\n[Trial {trial.number}] Running ({trial_label}):")
        for k, v in sorted(sampled_params.items()):
            if len(grid[k]) > 1:
                print(f"  * {k:22s}: {v} (sampled)")
            else:
                print(f"  * {k:22s}: {v} (fixed)")

        return run_training_subprocess(
            trial=trial,
            config_content=generated_config,
            checkpoint_manager=checkpoint_manager,
            config_prefix="optuna",
            verbose=args.verbose,
        )

    return objective


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization (HPO) for SASRec & gSASRec")
    parser.add_argument('--dataset', type=str, default='amazon_gts', help="Target dataset name (default: amazon_gts)")
    parser.add_argument('--model_type', type=str, default='gsasrec', choices=['sasrec', 'gsasrec'],
                        help="Model to tune: 'gsasrec' (G-BCE with multiple negatives) or 'sasrec'")
    parser.add_argument('--n_trials', type=int, default=15, help="Number of Optuna trials to run")
    parser.add_argument('--max_epochs', type=int, default=10000, help="Maximum epochs per trial (default: 10000)")
    parser.add_argument('--patience', type=int, default=200, help="Early stopping patience in epochs (default: 200)")
    parser.add_argument('--storage', type=str, default=None, 
                        help="Optuna DB storage URL (default: sqlite:///hpo_<dataset>_<model_type>.db)")
    parser.add_argument('--study_name', type=str, default=None, 
                        help="Optuna study name (default: hpo_<dataset>_<model_type>)")
    parser.add_argument('--csv_output', type=str, default=None,
                        help="Path for exported CSV results (default: hpo_results_<dataset>_<model_type>.csv)")
    parser.add_argument('--verbose', action='store_true',
                        help="Stream epoch-by-epoch training logs live to stdout")
    args = parser.parse_args()

    if args.storage is None:
        args.storage = f"sqlite:///hpo_{args.dataset}_{args.model_type}.db"
    if args.study_name is None:
        args.study_name = f"hpo_{args.dataset}_{args.model_type}"
    if args.csv_output is None:
        args.csv_output = f"hpo_results_{args.dataset}_{args.model_type}.csv"

    print("=" * 80)
    print(f"STARTING HPO FOR: {args.dataset.upper()} | MODEL: {args.model_type.upper()}")
    print(f"Validation Metric:       {TARGET_METRIC_LABEL}")
    print(f"Trials: {args.n_trials} | Max Epochs: {args.max_epochs} | Patience: {args.patience}")
    print(f"Storage: {args.storage} | Study: {args.study_name}")
    print(f"CSV Output:              {args.csv_output}")
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
    print_leaderboard(study, top_n=15, title="HYPERPARAMETER TUNING LEADERBOARD")
    export_to_csv(study, args.csv_output)

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

    print_parameter_importances(study)
    print("=" * 80)


if __name__ == "__main__":
    main()
