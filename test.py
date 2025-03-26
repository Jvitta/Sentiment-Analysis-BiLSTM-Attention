import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

from models.lstm_with_attention import BiLSTMAttention


def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # Extract text and labels from batch
            text, labels = batch
            text = text.to(device)
            labels = labels.to(device)
            
            # Calculate lengths and create mask
            lengths = torch.sum(text != 0, dim=1).to(device)
            mask = (text != 0).to(device)
            
            # Forward pass
            predictions, attention_weights = model(text, lengths, mask)
            loss = criterion(predictions, labels)
            
            # Calculate metrics
            total_loss += loss.item()
            binary_preds = (predictions > 0).float()
            
            # Convert to numpy for metrics calculation
            all_preds.append(binary_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Flatten lists to create 1D arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    print(f"Collected {len(all_preds)} predictions and {len(all_labels)} labels")
    print(f"Labels distribution: {np.bincount(all_labels.astype(int))}")
    print(f"Predictions distribution: {np.bincount(all_preds.astype(int))}")
    
    # Calculate metrics
    test_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    return test_loss, accuracy, precision, recall, f1, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    
    # Calculate percentages for display
    total = np.sum(cm)
    cm_percent = cm / total * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Add percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'{cm_percent[i, j]:.1f}%', 
                    ha='center', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_class_distribution(labels, title='Class Distribution', save_path='class_distribution.png'):
    """Plot class distribution of a dataset."""
    plt.figure(figsize=(8, 6))
    
    # Count class occurrences
    counts = np.bincount(labels.astype(int))
    
    # Create bar chart
    sns.barplot(x=['Negative (0)', 'Positive (1)'], y=counts)
    
    # Add text annotations with counts and percentages
    total = len(labels)
    for i, count in enumerate(counts):
        percentage = count / total * 100
        plt.text(i, count/2, f'{count}\n({percentage:.1f}%)', 
                 ha='center', va='center', fontsize=12, color='white')
    
    # Add labels and title
    plt.xlabel('Sentiment Class')
    plt.ylabel('Count')
    plt.title(title)
    
    # Save and close
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Class distribution plot saved to {save_path}")
    
    # Return class distribution info
    return {
        'negative': int(counts[0]), 
        'positive': int(counts[1]), 
        'total': total,
        'negative_percent': float(counts[0]/total*100),
        'positive_percent': float(counts[1]/total*100)
    }


def main():
    parser = argparse.ArgumentParser(description='Test model on test set')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/20250325_174141/model_epoch_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--output-dir', type=str, default='visualizations', 
                       help='Directory to save output plots and results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    X_test = np.load('data/processed_data/X_test.npy')
    y_test = np.load('data/processed_data/y_test.npy')
    print(f"Test set size: {len(X_test)}")
    
    # Plot class distribution
    plot_class_distribution(y_test, title='Test Set Class Distribution',
                           save_path=f'{args.output_dir}/test_class_distribution.png')
    
    # Create DataLoader
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config from bayesian_best_config.json
    print("Loading configuration from configs/bayesian_best_config.json")
    with open('configs/bayesian_best_config.json', 'r') as f:
        config = json.load(f)
    
    # Also load model_config for vocabulary size and embedding dimension
    with open('configs/model_config.json', 'r') as f:
        model_config = json.load(f)
    
    # Add necessary parameters from model_config to our config
    config['vocab_size'] = model_config['vocab_size']
    config['embedding_dim'] = model_config['embedding_dim']
    
    print(f"Config loaded: {config}")
    
    # Create model with the same config
    model = BiLSTMAttention(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy, test_precision, test_recall, test_f1, all_preds, all_labels = evaluate_model(
        model, test_loader, device
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, 
                         save_path=f'{args.output_dir}/confusion_matrix.png')
    
    # Print test results
    print("\n" + "="*60)
    print("Test Results:")
    print(f"Test Loss:       {test_loss:.6f}")
    print(f"Test Accuracy:   {test_accuracy:.6f}")
    print(f"Test Precision:  {test_precision:.6f}")
    print(f"Test Recall:     {test_recall:.6f}")
    print(f"Test F1 Score:   {test_f1:.6f}")
    
    # Save results to file
    result_path = os.path.join(args.output_dir, 'test_results.json')
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'checkpoint_used': checkpoint_path,
        'model_config': config
    }
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nTest results saved to {result_path}")
    print("="*60)


if __name__ == "__main__":
    main() 