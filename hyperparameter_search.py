import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import train
import argparse
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from torch_lr_finder import LRFinder
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from models.lstm_with_attention import BiLSTMAttention
import pickle

def plot_top_configs(top_results):
    """
    Plot comparison of top hyperparameter configurations
    
    Args:
        top_results: List of top configurations and their results
    """
    plt.figure(figsize=(12, 6))
    
    # Bar chart of mean validation accuracies
    plt.subplot(1, 2, 1)
    accuracies = [result['mean_val_acc'] for result in top_results]
    errors = [result['std_val_acc'] for result in top_results]
    config_labels = [f"Config {i+1}" for i in range(len(top_results))]
    
    plt.bar(config_labels, accuracies, yerr=errors, capsize=10)
    plt.ylim(max(0, min(accuracies) - 0.1), min(1.0, max(accuracies) + 0.1))
    plt.title('Validation Accuracy by Configuration')
    plt.ylabel('Mean Validation Accuracy')
    plt.xlabel('Configuration')
    
    # Add configuration details as text
    plt.subplot(1, 2, 2)
    plt.axis('off')
    config_text = "Top Configurations:\n\n"
    
    for i, result in enumerate(top_results):
        config_text += f"Config {i+1}:\n"
        for key, value in result['config'].items():
            if key != 'num_epochs':  # Skip epochs as it's fixed for hyperparameter search
                config_text += f"  {key}: {value}\n"
        config_text += f"  Accuracy: {result['mean_val_acc']:.4f} ± {result['std_val_acc']:.4f}\n\n"
    
    plt.text(0, 0.5, config_text, fontsize=9, verticalalignment='center')
    
    plt.tight_layout()
    os.makedirs('visualizations', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'visualizations/hyperparameter_comparison_{timestamp}.png')
    
    # Log to MLflow if in active run
    if mlflow.active_run():
        mlflow.log_artifact(f'visualizations/hyperparameter_comparison_{timestamp}.png')


def get_optimal_lr(lr_finder):
    """
    Get the optimal learning rate from the lr_finder history.
    This is a replacement for the non-existent suggestion() method.
    
    The approach finds the point with the steepest downward slope
    in the loss vs. learning rate curve.
    
    Args:
        lr_finder: The LRFinder object after running range_test
        
    Returns:
        float: The suggested optimal learning rate
    """
    lrs = lr_finder.history["lr"]
    losses = lr_finder.history["loss"]
    
    # Handle empty history
    if not lrs or not losses:
        print("Warning: Empty learning rate history. Using default learning rate.")
        return 1e-4  # Changed from 1e-3 to 1e-4 for sentiment analysis
    
    # Skip the beginning and end of the curve for more stable results
    skip_start = min(10, len(lrs) // 10)
    skip_end = min(5, len(lrs) // 20)
    
    if skip_start >= len(lrs) or skip_end >= len(lrs) or skip_start + skip_end >= len(lrs):
        # If not enough data points, use a simpler approach
        if len(lrs) > 3:
            min_loss_idx = losses.index(min(losses))
            # Return the learning rate at minimum loss or slightly before
            return lrs[max(0, min_loss_idx - 1)]
        return lrs[0] if lrs else 1e-4  # Changed from 1e-3 to 1e-4 for sentiment analysis
    
    # Calculate gradients with safeguards for division by zero
    gradients = []
    for i in range(skip_start, len(lrs) - skip_end - 1):
        lr_diff = lrs[i + 1] - lrs[i]
        if abs(lr_diff) < 1e-10:  # Avoid division by near-zero
            continue
        gradients.append((losses[i + 1] - losses[i]) / lr_diff)
    
    # Check if we have valid gradients
    if not gradients:
        print("Warning: Could not calculate valid gradients. Using median learning rate.")
        return lrs[len(lrs) // 2]
    
    # Find the point with the steepest negative gradient
    # (use smoothed gradient to avoid noise)
    smooth_window = min(5, len(gradients) // 5)
    if smooth_window > 0 and len(gradients) > smooth_window:
        smoothed_gradients = []
        for i in range(len(gradients) - smooth_window + 1):
            smoothed_gradients.append(sum(gradients[i:i+smooth_window]) / smooth_window)
        
        if not smoothed_gradients:
            # If we somehow ended up with no smoothed gradients
            steepest_idx = gradients.index(min(gradients))
        else:
            steepest_idx = smoothed_gradients.index(min(smoothed_gradients))
    else:
        steepest_idx = gradients.index(min(gradients))
    
    # Return the learning rate at the steepest point
    suggested_lr = lrs[skip_start + steepest_idx]
    
    return suggested_lr

def find_learning_rate(data_path, batch_size=32, min_lr=1e-7, max_lr=10, num_iter=100):
    """
    Use the LR Finder to discover the optimal learning rate for the model.
    
    Args:
        data_path: Path to preprocessed data
        batch_size: Batch size to use for training
        min_lr: Minimum learning rate to explore
        max_lr: Maximum learning rate to explore
        num_iter: Number of iterations for the LR finder
    
    Returns:
        suggested_lr: The suggested learning rate based on the finder
    """
    print("Starting Learning Rate Finder...")
    
    # Set up MLflow if available
    try:
        experiment_name = "sentiment_analysis_lr_finder"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = mlflow.start_run(run_name=f"lr_finder_{timestamp}").info.run_id
    except Exception as e:
        print(f"Warning: Could not set up MLflow: {e}")
        run_id = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data and config
    with open(os.path.join(data_path, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_path, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(data_path, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)
    
    # Debug prints for raw data
    print("Raw data shapes:")
    print(f"Number of training sequences: {len(train_data['X'])}")
    print(f"Number of validation sequences: {len(val_data['X'])}")
    
    # Debug y_train data
    print("\nInspecting y_train data:")
    print(f"Type of y_train: {type(train_data['y'])}")
    print(f"First 5 y_train values: {train_data['y'][:5]}")
    print(f"Shape of y_train if numpy: {train_data['y'].shape if hasattr(train_data['y'], 'shape') else 'not a numpy array'}")
    
    # Debug X_train data
    print("\nInspecting X_train data:")
    print(f"Type of X_train: {type(train_data['X'])}")
    print(f"Length of X_train: {len(train_data['X'])}")
    print(f"First sequence shape: {np.array(train_data['X'][0]).shape}")
    print(f"First sequence: {train_data['X'][0]}")
    
    # Check if all X sequences have the same length
    sequence_lengths = [len(seq) for seq in train_data['X']]
    if len(set(sequence_lengths)) > 1:
        print("\nWarning: Found varying sequence lengths:")
        for length in sorted(set(sequence_lengths)):
            count = sequence_lengths.count(length)
            print(f"Length {length}: {count} sequences")
    
    # Detailed inspection of y_train values
    print("\nChecking for problematic y_train values...")
    for idx, val in enumerate(train_data['y']):
        if not isinstance(val, (int, np.int32, np.int64)) or val not in [0, 1]:
            print(f"Problematic value at index {idx}: {val} (type: {type(val)})")
    
    # Convert to numpy arrays
    print("\nAttempting to convert to numpy arrays...")
    X_train = np.array(train_data['X'])
    X_val = np.array(val_data['X'])
    y_train = np.array(train_data['y'])
    y_val = np.array(val_data['y'])
    
    # Print tensor shapes
    print("\nTensor shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.float)
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttention(
        vocab_size=config['vocab_size'],  # Get vocab size from config
        embedding_dim=config['embedding_dim'],  # Get embedding dim from config
        hidden_dim=256,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    # Initialize optimizer and loss
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    # Initialize LR Finder with custom forward function
    def custom_forward(model, x, y):
        # Ensure input is properly padded to max_seq_length
        if x.size(1) > 26:
            x = x[:, :26]
        return model(x)
    
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.forward = custom_forward
    
    # Run LR Finder
    print(f"Running LR Finder from {min_lr} to {max_lr} over {num_iter} iterations...")
    lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=max_lr, num_iter=num_iter, step_mode="exp")
    
    # Get suggestion
    suggested_lr = get_optimal_lr(lr_finder)
    print(f"Suggested learning rate: {suggested_lr:.8f}")
    
    # Plot the results
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        lr_finder.plot(ax=ax, skip_start=10, skip_end=5)
        ax.set_title('Learning Rate Finder')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.axvline(x=suggested_lr, color='r', linestyle='--', alpha=0.7)
        
        # Save the plot
        os.makedirs('visualizations', exist_ok=True)
        plot_path = f'visualizations/lr_finder_{timestamp}.png'
        plt.savefig(plot_path)
        plt.close()
        
        # Log to MLflow
        if run_id:
            try:
                mlflow.log_metric("suggested_lr", suggested_lr)
                mlflow.log_artifact(plot_path)
            except Exception as e:
                print(f"Warning: Could not log to MLflow: {e}")
    except Exception as e:
        print(f"Warning: Error creating learning rate plot: {e}")
    
    # Save the suggested learning rate to a config file
    lr_config = {
        'batch_size': batch_size,
        'learning_rate': suggested_lr,
        'weight_decay': 1e-5,
        'use_lr_scheduler': True,
        'num_epochs': 50
    }
    
    os.makedirs('configs', exist_ok=True)
    config_path = os.path.join('configs', f'lr_finder_config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(lr_config, f, indent=2)
    
    print(f"Configuration with suggested learning rate saved to {config_path}")
    
    # Also save to a standard name for easy reference
    standard_config_path = os.path.join('configs', 'lr_finder_config.json')
    with open(standard_config_path, 'w') as f:
        json.dump(lr_config, f, indent=2)
    
    print(f"Configuration also saved to {standard_config_path}")
    
    # Log to MLflow if available
    if run_id:
        try:
            mlflow.log_artifact(config_path)
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")
    
    return suggested_lr

def bayesian_search(data_path, num_folds=4, test_fold=5, n_trials=25, timeout=None):
    """
    Perform Bayesian hyperparameter optimization using Optuna
    
    Args:
        data_path: Path to preprocessed data
        num_folds: Number of folds to use for cross-validation
        test_fold: Fold to reserve for final testing
        n_trials: Number of trials to run
        timeout: Stop study after the given number of seconds (None means no timeout)
    
    Returns:
        best_config: Configuration that achieved best validation performance
        study: The completed Optuna study object
    """
    print("Starting Bayesian hyperparameter optimization...")
    
    # Set up MLflow experiment
    experiment_name = "esc50_bayesian_optimization"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    # Define the objective function for Optuna
    def objective(trial):
        # Define hyperparameters to optimize
        config = {
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
            'learning_rate': 0.00018096,  # Fixed to the optimal value found by LR finder
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'use_lr_scheduler': trial.suggest_categorical('use_lr_scheduler', [True, False]),
            'num_epochs': 10  # Fixed for optimization
        }
        
        # Optional: Add more hyperparameters for better tuning
        if trial.suggest_categorical('use_dropout', [True, False]):
            config['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        # Optimize scheduling parameters if scheduler is used
        if config['use_lr_scheduler']:
            config['scheduler_patience'] = trial.suggest_int('scheduler_patience', 3, 10)
            config['scheduler_factor'] = trial.suggest_float('scheduler_factor', 0.1, 0.5)
        
        # Add data augmentation parameters to test their impact
        config['use_augmentation'] = trial.suggest_categorical('use_augmentation', [True, False])
        if config['use_augmentation']:
            config['aug_strength'] = trial.suggest_float('aug_strength', 0.1, 0.5)
        
        trial_id = trial.number
        
        # Track this configuration with MLflow
        with mlflow.start_run(run_name=f"trial_{trial_id}"):
            # Log parameters
            for key, value in config.items():
                mlflow.log_param(key, value)
            
            # Evaluate this configuration with cross-validation
            _, cv_results = cross_validation(config, data_path, num_folds, test_fold)
            
            mean_val_acc = cv_results['mean_val_acc']
            
            # Log to MLflow
            mlflow.log_metric("mean_val_acc", mean_val_acc)
            mlflow.log_metric("std_val_acc", cv_results['std_val_acc'])
            
            # Report intermediate values for pruning
            trial.report(mean_val_acc, step=1)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return mean_val_acc
    
    # Create a study object and optimize the objective function
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    sampler = TPESampler(seed=42)  # Use TPE algorithm with a fixed seed for reproducibility
    
    study = optuna.create_study(
        direction="maximize",  # We want to maximize validation accuracy
        sampler=sampler,
        pruner=pruner,
        study_name="esc50_optimization"
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print("\n===== Bayesian Optimization Results =====")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    
    # Create best config dictionary
    best_config = study.best_params.copy()
    
    # Add some required keys that might not be in the params
    if 'use_dropout' in best_config and best_config['use_dropout']:
        # If dropout is enabled, keep the dropout rate
        pass
    else:
        # Otherwise remove the dropout flag from the final config
        if 'use_dropout' in best_config:
            del best_config['use_dropout']
    
    # For final training, we want to use more epochs
    best_config['num_epochs'] = 50
    
    # Additional scheduler parameters if present
    scheduler_keys = ['scheduler_patience', 'scheduler_factor']
    for key in scheduler_keys:
        if key not in best_config and best_config.get('use_lr_scheduler', False):
            if key == 'scheduler_patience':
                best_config[key] = 5
            elif key == 'scheduler_factor':
                best_config[key] = 0.5
    
    print(json.dumps(best_config, indent=2))
    
    # Save best configuration for later use
    os.makedirs('configs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"bayesian_best_config_{timestamp}.json"
    config_path = os.path.join('configs', config_filename)
    
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Also save a generic best_config.json for easy reference
    with open(os.path.join('configs', 'bayesian_best_config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Save study statistics
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image('visualizations/bayesian_optimization_history.png')
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image('visualizations/bayesian_param_importances.png')
        
        # Log to MLflow if in active run
        if mlflow.active_run():
            mlflow.log_artifact('visualizations/bayesian_optimization_history.png')
            mlflow.log_artifact('visualizations/bayesian_param_importances.png')
    except:
        print("Warning: Could not create optimization visualizations. This may be due to missing plotly or other visualization dependencies.")
    
    return best_config, study

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter search for sentiment analysis')
    parser.add_argument('--find-lr', action='store_true',
                      help='Run learning rate finder to determine optimal learning rate range')
    parser.add_argument('--min-lr', type=float, default=1e-7,
                      help='Minimum learning rate to explore (default: 1e-7)')
    parser.add_argument('--max-lr', type=float, default=10.0,
                      help='Maximum learning rate to explore (default: 10.0)')
    parser.add_argument('--num-iter', type=int, default=100,
                      help='Number of iterations for LR finder (default: 100, higher values give smoother curves)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for LR finder (default: 32)')
    parser.add_argument('--bayesian', action='store_true',
                      help='Use Bayesian optimization for hyperparameter search')
    parser.add_argument('--n-trials', type=int, default=25,
                      help='Number of trials for Bayesian optimization (default: 25)')
    parser.add_argument('--timeout', type=int, default=None,
                      help='Timeout in seconds for Bayesian optimization (default: None)')
    args = parser.parse_args()
    
    # Data path
    data_path = 'data/processed_data'
    
    # Run learning rate finder if requested
    if args.find_lr:
        print(f"Starting learning rate finder with batch size {args.batch_size}")
        print(f"Learning rate range: {args.min_lr} to {args.max_lr} over {args.num_iter} iterations")
        try:
            suggested_lr = find_learning_rate(
                data_path,
                batch_size=args.batch_size,
                min_lr=args.min_lr,
                max_lr=args.max_lr,
                num_iter=args.num_iter
            )
            print(f"Learning rate finder completed successfully!")
            print(f"Suggested learning rate: {suggested_lr:.8f}")
            print(f"Configuration saved to configs/lr_finder_config.json")
        except Exception as e:
            print(f"Error running learning rate finder: {e}")
        return
    
    # Use Bayesian optimization if requested
    if args.bayesian:
        print("Using Bayesian optimization for hyperparameter search")
        best_config, _ = bayesian_search(
            data_path,
            n_trials=args.n_trials,
            timeout=args.timeout
        )
        print("\nTo train with the best configuration, run:")
        print("python train.py --config configs/bayesian_best_config.json --mode final")
        return
    
    # If no arguments provided, show help
    parser.print_help()

if __name__ == "__main__":
    main()
