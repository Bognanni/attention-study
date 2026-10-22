import argparse
import optuna
from optuna.samplers import GridSampler

from hpo_utils import (
    TEMPLATE_CONFIG,
    TARGET_METRIC_LABEL,
    get_search_grid,
    calculate_total_combinations,
    BestCheckpointManager,
    print_leaderboard,
    export_to_csv,
    print_parameter_importances,
    run_training_subprocess,
)


def create_objective(args, grid: dict, total_combinations: int, checkpoint_manager: BestCheckpointManager):
    """
    Builds the objective function for Exhaustive Grid Search.
    """
    def objective(trial):
        # Sample the exact combination assigned by GridSampler
        sampled_params = {}
        for param_name, values_list in grid.items():
            sampled_params[param_name] = trial.suggest_categorical(param_name, values_list)

        # Generate configuration content
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
        )

        trial_idx = trial.number + 1
        print("\n" + "-" * 80)
        print(f"[Grid Trial {trial_idx}/{total_combinations}] Testing configuration ({args.model_type.upper()}):")
        for k, v in sorted(sampled_params.items()):
            print(f"  * {k:22s}: {v}")
        print("-" * 80)

        return run_training_subprocess(
            trial=trial,
            config_content=generated_config,
            checkpoint_manager=checkpoint_manager,
            config_prefix="grid",
            verbose=args.verbose,
        )

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive Grid Search for SASRec and gSASRec (Full Cartesian Product)"
    )
    parser.add_argument('--dataset', type=str, default='amazon_gts',
                        help="Target dataset name (default: amazon_gts)")
    parser.add_argument('--model_type', type=str, default='gsasrec', choices=['sasrec', 'gsasrec'],
                        help="Model type: 'sasrec' or 'gsasrec' (default: gsasrec)")
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

    # Retrieve unified search grid for model type
    grid = get_search_grid(args.model_type)
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
    print(f"Validation Metric:         {TARGET_METRIC_LABEL}")
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
        print_leaderboard(study, top_n=15, title="GRID SEARCH LEADERBOARD")
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
    print_leaderboard(study, top_n=15, title="GRID SEARCH LEADERBOARD")
    export_to_csv(study, args.csv_output)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        best_trial = study.best_trial
        print(f"\nBEST OVERALL RESULT:")
        print(f"  * Best Trial:            #{best_trial.number}")
        print(f"  * Best Validation Score:  {best_trial.value:.6f}")
        print(f"  * Preserved Checkpoint:   {checkpoint_manager.best_checkpoint_path}")
        print("  * Best Hyperparameters:")
        for param, val in best_trial.params.items():
            print(f"    - {param}: {val}")

    print_parameter_importances(study)
    print("=" * 80)


if __name__ == "__main__":
    main()
