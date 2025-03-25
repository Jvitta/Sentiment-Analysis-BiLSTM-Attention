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

from data_loader import create_dataloaders
from models.lstm_with_attention import BiLSTMAttention

# Initialize MLflow
mlflow.set_tracking_uri("file:./mlruns")  # Store runs locally
mlflow.set_experiment("sentiment_analysis")

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, learning_rate=0.001, weight_decay=1e-5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 5
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
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": 0.5,
            "scheduler_patience": 2,
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
            'val_accuracies': self.val_accuracies
        }
        
        # Save regular checkpoint
        checkpoint_path = f'{self.checkpoint_dir}/model_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if validation loss is better
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_checkpoint_path = f'{self.checkpoint_dir}/model_epoch_best.pt'
            torch.save(checkpoint, best_checkpoint_path)
            
            # Log best model to MLflow
            mlflow.log_artifact(best_checkpoint_path, "best_model")
    
    def plot_metrics(self):
        """Plot training and validation metrics."""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.title('Accuracies')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        metrics_plot_path = 'visualizations/training_metrics.png'
        plt.savefig(metrics_plot_path)
        plt.close()
        
        # Log metrics plot to MLflow
        mlflow.log_artifact(metrics_plot_path, "metrics_plots")
    
    def train(self, num_epochs=10):
        """Train the model."""
        print(f"Training on device: {self.device}")
        
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
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Plot metrics
        self.plot_metrics()
        
        # Log final metrics
        mlflow.log_metrics({
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "final_train_acc": self.train_accuracies[-1],
            "final_val_acc": self.val_accuracies[-1]
        })
        
        # End MLflow run
        mlflow.end_run()


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load preprocessed data from NumPy files
    X_train = np.load('data/processed_data/X_train.npy')
    X_val = np.load('data/processed_data/X_val.npy')
    X_test = np.load('data/processed_data/X_test.npy')
    y_train = np.load('data/processed_data/y_train.npy')
    y_val = np.load('data/processed_data/y_val.npy')
    y_test = np.load('data/processed_data/y_test.npy')
    embedding_matrix = np.load('data/processed_data/embedding_matrix.npy')
    
    # Load config from JSON
    with open('configs/model_config.json', 'r') as f:
        config = json.load(f)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train.tolist(), X_val.tolist(), X_test.tolist(),
        y_train.tolist(), y_val.tolist(), y_test.tolist(),
        batch_size=32
    )
    
    # Initialize model
    model = BiLSTMAttention(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=256,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    # Load pre-trained embeddings
    model.load_pretrained_embeddings(embedding_matrix)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.001
    )
    
    # Train model
    trainer.train(num_epochs=10)


if __name__ == "__main__":
    main()
