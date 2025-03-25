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
from data_loader import create_dataloaders

def plot_top_configs(top_results):
    """Plot comparative results of top hyperparameter configurations."""
    if not top_results:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Group by learning rate
    lrs = []
    val_accs = []
    
    for config, metrics in top_results:
        lrs.append(config['learning_rate'])
        val_accs.append(metrics['val_accuracy'])
    
    # Sort by learning rate
    lr_acc_pairs = sorted(zip(lrs, val_accs))
    lrs, val_accs = zip(*lr_acc_pairs)
    
    # Plot learning rate vs validation accuracy
    plt.plot(lrs, val_accs, 'o-', label='Validation Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.xscale('log')
    plt.title('Learning Rate vs Validation Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Mark best point
    best_idx = val_accs.index(max(val_accs))
    plt.scatter([lrs[best_idx]], [val_accs[best_idx]], c='red', s=100, label=f'Best LR: {lrs[best_idx]:.8f}')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(f'visualizations/hyperparameter_comparison_{timestamp}.png')
    plt.close()
    
    # Log to MLflow if available
    try:
        mlflow.log_artifact(f'visualizations/hyperparameter_comparison_{timestamp}.png')
    except:
        pass


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
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    X_val = np.load(os.path.join(data_path, 'X_val.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))
    
    # Load config from JSON
    config_path = os.path.join('configs', 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.float)
    
    # Create data loaders
    train_loader, val_loader, _ = create_dataloaders(
        X_train.tolist(), X_val.tolist(), [],  # Empty test set since we don't need it
        y_train.tolist(), y_val.tolist(), [],
        batch_size=batch_size
    )
    
    # Initialize model
    # For learning rate finder, using CPU can be more stable
    device = torch.device("cpu")
    print(f"Using {device} for learning rate finder for better stability")
    
    model = BiLSTMAttention(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=256,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    # Initialize optimizer and loss
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss instead of BCELoss
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    # Create a simple adapter for our model during LR finding
    class ModelAdapter(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, batch):
            # Handle batch from DataLoader which is a tuple (text, lengths, labels)
            text, lengths = batch[0], batch[1]
            
            # Debug zero lengths
            zero_lengths = (lengths == 0).sum().item()
            if zero_lengths > 0:
                print(f"Warning: Found {zero_lengths} sequences with zero length")
                print(f"First few lengths: {lengths[:10]}")
                print(f"Full length tensor shape: {lengths.shape}")
                # Ensure all lengths are at least 1 to avoid LSTM errors
                lengths = torch.clamp(lengths, min=1)
            
            # Create mask
            mask = (text != 0)
            
            # Get model prediction - our model returns (output, attention_weights) tuple
            outputs, _ = self.model(text, lengths, mask)
            
            # Return just the first element (logits)
            return outputs
    
    # Wrap our model in the adapter
    model_adapter = ModelAdapter(model)
    
    # Use the standard LRFinder with our adapter
    lr_finder = LRFinder(model_adapter, optimizer, criterion, device=device)
    
    # Run LR Finder
    print(f"Running LR Finder from {min_lr} to {max_lr} over {num_iter} iterations...")
    
    try:
        # Run LR finder
        lr_finder.range_test(
            train_loader, 
            val_loader=val_loader, 
            end_lr=min(0.1, max_lr),  # Use a more conservative upper bound (0.1)
            num_iter=min(50, num_iter),  # Reduce number of iterations for stability
            step_mode="exp",
            diverge_th=5.0  # Increase divergence threshold to avoid early stopping
        )
        
        # Get suggestion
        if len(lr_finder.history['lr']) > 10:
            # Find learning rate with steepest gradient
            suggested_lr = lr_finder.steepest_gradient(skip_start=5, skip_end=5)
            if suggested_lr is None:
                # If steepest gradient fails, use minimum loss
                suggested_lr = lr_finder.get_best_lr(skip_start=5, skip_end=5)
            print(f"Suggested learning rate: {suggested_lr:.8f}")
        else:
            # If we don't have enough data points, use a reasonable default
            suggested_lr = 1e-4
            print(f"Not enough data points in LR finder history. Using default: {suggested_lr}")
        
    except Exception as e:
        import traceback
        print(f"Error running learning rate finder: {e}")
        print("Full traceback:")
        traceback.print_exc()
        suggested_lr = 1e-4  # Return a reasonable default
        print(f"Using default learning rate: {suggested_lr}")
    
    # Plot the results
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        lr_finder.plot(ax=ax, skip_start=5, skip_end=5)
        ax.set_title('Learning Rate Finder')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.axvline(x=suggested_lr, color='r', linestyle='--', alpha=0.7, label='steepest gradient')
        plt.legend()
        
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
            import traceback
            print(f"Error running learning rate finder: {e}")
            print("Full traceback:")
            traceback.print_exc()
            
            # Create a default learning rate configuration even if finder fails
            print("\nCreating default learning rate configuration...")
            default_lr = 1e-4  # A reasonable default for sentiment analysis
            
            # Save the default learning rate to a config file
            lr_config = {
                'batch_size': args.batch_size,
                'learning_rate': default_lr,
                'weight_decay': 1e-5,
                'use_lr_scheduler': True,
                'num_epochs': 50
            }
            
            os.makedirs('configs', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_path = os.path.join('configs', f'lr_finder_config_{timestamp}.json')
            with open(config_path, 'w') as f:
                json.dump(lr_config, f, indent=2)
            
            # Also save to a standard name for easy reference
            standard_config_path = os.path.join('configs', 'lr_finder_config.json')
            with open(standard_config_path, 'w') as f:
                json.dump(lr_config, f, indent=2)
            
            print(f"Default configuration with learning rate {default_lr} saved to {standard_config_path}")
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
