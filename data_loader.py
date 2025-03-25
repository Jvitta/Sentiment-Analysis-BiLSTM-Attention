import torch
from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    """Simplified dataset class for pre-padded sequences."""
    def __init__(self, sequences, labels, pad_idx=0):
        # Convert to tensors once at initialization
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.lengths = torch.sum(self.sequences != pad_idx, dim=1).long()
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32, pad_idx=0):
    """Create DataLoaders for train, validation and test sets."""
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, pad_idx)
    val_dataset = SentimentDataset(X_val, y_val, pad_idx)
    test_dataset = SentimentDataset(X_test, y_test, pad_idx)
    
    # Create data loaders with default collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader 