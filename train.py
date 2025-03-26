import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import json
import argparse

from data_loader import create_dataloaders
from models.lstm_with_attention import BiLSTMAttention

# Initialize MLflow
mlflow.set_tracking_uri("file:./mlruns")  # Store runs locally
mlflow.set_experiment("sentiment_analysis")

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, learning_rate=0.001, weight_decay=1e-5, 
                 use_lr_scheduler=True, scheduler_patience=2, scheduler_factor=0.5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        if use_lr_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=scheduler_factor,
                patience=scheduler_patience, 
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.optimizer_lr_history = []  # Track learning rate throughout training
        
        # Best metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_precision = 0.0
        self.best_val_recall = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = -1
        
        # Early stopping
        self.patience = 10
        self.patience_counter = 0
        
        # Create checkpoints directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = f'checkpoints/{self.timestamp}'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Start MLflow run
        self.run = mlflow.start_run()
        
        # Log model parameters
        model_params = {
            "vocab_size": model.embedding.num_embeddings,
            "embedding_dim": model.embedding.embedding_dim,
            "hidden_dim": model.lstm.hidden_size,
            "num_layers": model.lstm.num_layers,
            "dropout": model.dropout.p,
            "bidirectional": model.lstm.bidirectional,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": "Adam",
            "use_lr_scheduler": use_lr_scheduler,
            "scheduler_type": "ReduceLROnPlateau" if use_lr_scheduler else "None",
            "scheduler_factor": scheduler_factor if use_lr_scheduler else None,
            "scheduler_patience": scheduler_patience if use_lr_scheduler else None,
            "early_stopping_patience": self.patience,
            "device": str(device)
        }
        mlflow.log_params(model_params)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.train_loader, desc='Training'):
            text, lengths, labels = batch
            text = text.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)
            
            # Create mask for attention
            mask = (text != 0).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, _ = self.model(text, lengths, mask)
            loss = self.criterion(predictions, labels)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            predictions = (predictions > 0.5).float()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, accuracy
    
    def evaluate(self, loader, desc='Evaluating'):
        """Evaluate the model on a data loader."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                text, lengths, labels = batch
                text = text.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                
                # Create mask for attention
                mask = (text != 0).to(self.device)
                
                # Forward pass
                predictions, _ = self.model(text, lengths, mask)
                loss = self.criterion(predictions, labels)
                
                # Calculate metrics
                total_loss += loss.item()
                predictions = (predictions > 0.5).float()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        return epoch_loss, accuracy, precision, recall, f1
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'optimizer_lr_history': self.optimizer_lr_history,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_precision': self.best_val_precision,
            'best_val_recall': self.best_val_recall,
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch
        }
        
        # Save regular checkpoint
        checkpoint_path = f'{self.checkpoint_dir}/model_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if validation loss is better
        if val_loss <= self.best_val_loss:
            best_checkpoint_path = f'{self.checkpoint_dir}/model_epoch_best.pt'
            torch.save(checkpoint, best_checkpoint_path)
            
            # Log best model to MLflow
            mlflow.log_artifact(best_checkpoint_path, "best_model")
    
    def plot_metrics(self):
        """Plot training and validation metrics."""
        # Create directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # Track learning rate changes
        lr_changes = []
        lr_values = []
        
        # Only process learning rate history if we have enough epochs
        if len(self.optimizer_lr_history) > 1:
            current_lr = self.optimizer_lr_history[0]
            lr_changes.append(0)
            lr_values.append(current_lr)
            
            for i in range(1, len(self.optimizer_lr_history)):
                if self.optimizer_lr_history[i] != current_lr:
                    current_lr = self.optimizer_lr_history[i]
                    lr_changes.append(i)
                    lr_values.append(current_lr)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Losses')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Add vertical lines for learning rate changes on loss plot
        if len(self.train_losses) > 0 and len(lr_changes) > 1:
            max_loss = max(max(self.train_losses), max(self.val_losses))
            for i, epoch in enumerate(lr_changes):
                if i > 0:  # Skip the initial learning rate
                    ax1.axvline(x=epoch-1, color='r', linestyle='--', alpha=0.5)
                    ax1.text(epoch-1, max_loss*0.9, 
                            f'LR: {lr_values[i]:.2E}', rotation=90, verticalalignment='top')
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_title('Accuracies')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # Add vertical lines for learning rate changes on accuracy plot
        if len(self.train_accuracies) > 0 and len(lr_changes) > 1:
            min_acc = min(min(self.train_accuracies), min(self.val_accuracies))
            max_acc = max(max(self.train_accuracies), max(self.val_accuracies))
            for i, epoch in enumerate(lr_changes):
                if i > 0:  # Skip the initial learning rate
                    ax2.axvline(x=epoch-1, color='r', linestyle='--', alpha=0.5)
                    ax2.text(epoch-1, min_acc + (max_acc - min_acc)*0.1, 
                            f'LR: {lr_values[i]:.2E}', rotation=90, verticalalignment='bottom')
        
        plt.tight_layout()
        metrics_plot_path = f'visualizations/training_metrics_{self.timestamp}.png'
        plt.savefig(metrics_plot_path)
        
        # Also save with a standard name for easy reference
        plt.savefig('visualizations/training_metrics.png')
        plt.close()
        
        # Log metrics plot to MLflow
        mlflow.log_artifact(metrics_plot_path, "metrics_plots")
    
    def print_best_results(self):
        """Print a summary of the best results achieved during training."""
        print("\n" + "="*60)
        print(f"TRAINING COMPLETED - BEST RESULTS (Epoch {self.best_epoch+1})")
        print("="*60)
        print(f"Best Validation Loss:      {self.best_val_loss:.6f}")
        print(f"Best Validation Accuracy:  {self.best_val_acc:.6f} ({self.best_val_acc*100:.2f}%)")
        print(f"Best Validation Precision: {self.best_val_precision:.6f}")
        print(f"Best Validation Recall:    {self.best_val_recall:.6f}")
        print(f"Best Validation F1 Score:  {self.best_val_f1:.6f}")
        print("="*60)
        
        # Also log the best metrics summary as an artifact
        with open(f'{self.checkpoint_dir}/best_metrics_summary.txt', 'w') as f:
            f.write(f"Best Results (Epoch {self.best_epoch+1}):\n")
            f.write(f"Validation Loss:      {self.best_val_loss:.6f}\n")
            f.write(f"Validation Accuracy:  {self.best_val_acc:.6f} ({self.best_val_acc*100:.2f}%)\n")
            f.write(f"Validation Precision: {self.best_val_precision:.6f}\n")
            f.write(f"Validation Recall:    {self.best_val_recall:.6f}\n")
            f.write(f"Validation F1 Score:  {self.best_val_f1:.6f}\n")
        
        # Log the summary to MLflow
        mlflow.log_artifact(f'{self.checkpoint_dir}/best_metrics_summary.txt', "metrics")
    
    def train(self, num_epochs=10):
        """Train the model."""
        print(f"Training on device: {self.device}")
        
        # Record initial learning rate
        initial_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer_lr_history.append(initial_lr)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
            print(f"Learning Rate: {current_lr:.2E}")
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "learning_rate": current_lr
            }, step=epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Record the learning rate after scheduler may have changed it
            self.optimizer_lr_history.append(self.optimizer.param_groups[0]['lr'])
            
            # Track best performance and early stopping
            if val_acc > self.best_val_acc:
                # Using accuracy as primary metric for best model (instead of loss)
                print(f"✓ New best validation accuracy: {val_acc:.4f} (previous: {self.best_val_acc:.4f})")
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_val_precision = val_precision
                self.best_val_recall = val_recall
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
            # Else if val_loss is better but accuracy is not, still save checkpoint but don't reset patience
            elif val_loss < self.best_val_loss:
                print(f"✓ New best validation loss: {val_loss:.4f} (previous: {self.best_val_loss:.4f})")
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                self.patience_counter += 1
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Plot metrics
        self.plot_metrics()
        
        # Find the epoch with the best validation metrics
        best_acc_epoch = np.argmax(self.val_accuracies)
        best_val_acc = self.val_accuracies[best_acc_epoch]
        
        # Update best metrics if our tracking missed any
        if best_val_acc > self.best_val_acc:
            print(f"Found better accuracy in history: {best_val_acc:.4f} at epoch {best_acc_epoch+1}")
            self.best_val_acc = best_val_acc
            self.best_epoch = best_acc_epoch
        
        # Print best results
        self.print_best_results()
        
        # Log final metrics
        mlflow.log_metrics({
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_val_precision": self.best_val_precision,
            "best_val_recall": self.best_val_recall,
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "final_train_acc": self.train_accuracies[-1],
            "final_val_acc": self.val_accuracies[-1]
        })
        
        # End MLflow run
        mlflow.end_run()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--config', type=str, default='configs/bayesian_best_config.json',
                        help='Path to configuration file (default: configs/bayesian_best_config.json)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        train_config = json.load(f)
    
    print("Training with configuration:")
    print(json.dumps(train_config, indent=2))
    
    # Load preprocessed data from NumPy files
    X_train = np.load('data/processed_data/X_train.npy')
    X_val = np.load('data/processed_data/X_val.npy')
    X_test = np.load('data/processed_data/X_test.npy')
    y_train = np.load('data/processed_data/y_train.npy')
    y_val = np.load('data/processed_data/y_val.npy')
    y_test = np.load('data/processed_data/y_test.npy')
    embedding_matrix = np.load('data/processed_data/embedding_matrix.npy')
    
    # Load model config
    with open('configs/model_config.json', 'r') as f:
        model_config = json.load(f)
    
    # Create data loaders with batch size from config
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train.tolist(), X_val.tolist(), X_test.tolist(),
        y_train.tolist(), y_val.tolist(), y_test.tolist(),
        batch_size=train_config['batch_size']
    )
    
    # Initialize model with parameters from config
    model = BiLSTMAttention(
        vocab_size=model_config['vocab_size'],
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=train_config['hidden_dim'],
        num_layers=train_config['num_layers'],
        dropout=train_config['dropout']
    ).to(device)
    
    # Load pre-trained embeddings
    model.load_pretrained_embeddings(embedding_matrix)
    
    # Initialize trainer with learning rate and weight decay from config
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        use_lr_scheduler=train_config['use_lr_scheduler'],
        scheduler_patience=train_config['scheduler_patience'],
        scheduler_factor=train_config['scheduler_factor']
    )
    
    # Train model with number of epochs from config
    trainer.train(num_epochs=train_config['num_epochs'])


if __name__ == "__main__":
    main()
