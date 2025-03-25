import torch
import pickle
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data_loader import create_dataloaders
from models.lstm_with_attention import BiLSTMAttention


def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = torch.nn.BCELoss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            text, lengths, labels = batch
            text = text.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            # Create mask for attention
            mask = (text != 0).to(device)
            
            # Forward pass
            predictions, _ = model(text, lengths, mask)
            loss = criterion(predictions, labels)
            
            # Calculate metrics
            total_loss += loss.item()
            predictions = (predictions > 0.5).float()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    return test_loss, accuracy, precision, recall, f1


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load preprocessed data
    with open('data/processed_data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open('configs/model_config.json', 'r') as f:
        config = json.load(f)
    
    # Load embedding matrix
    embedding_matrix = np.load('data/processed_data/embedding_matrix.npy')
    
    # Load test data from NumPy files
    X_test = np.load('data/processed_data/X_test.npy')
    y_test = np.load('data/processed_data/y_test.npy')
    
    # Create test data loader
    _, _, test_loader = create_dataloaders(
        X_test.tolist(), [], [],  # Empty validation and train sets
        y_test.tolist(), [], [],
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
    
    # Load best model checkpoint
    checkpoint = torch.load('checkpoints/model_epoch_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(test_data['y'], model.predict(test_loader, device))


if __name__ == "__main__":
    main() 