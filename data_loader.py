import torch
from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    """Simplified dataset class for pre-padded sequences."""
    def __init__(self, sequences, labels, pad_idx=0):
        # Convert sequences to tensor and ensure proper shape
        if isinstance(sequences, torch.Tensor):
            self.sequences = sequences.long()  # Ensure long type for embeddings
        else:
            self.sequences = torch.tensor(sequences, dtype=torch.long)
        
        # Check sequences shape and ensure it's 2D
        if len(self.sequences.shape) != 2:
            raise ValueError(f"Expected sequences to be 2D tensor but got shape {self.sequences.shape}. "
                            "Sequences should have shape [batch_size, seq_length].")
        
        # Convert labels to tensor and ensure proper shape
        if isinstance(labels, torch.Tensor):
            self.labels = labels.float()  # Ensure float type for BCEWithLogitsLoss
        else:
            self.labels = torch.tensor(labels, dtype=torch.float)
        
        # Ensure labels is 1D (can be 1D or 2D with second dim = 1)
        if len(self.labels.shape) > 1:
            if self.labels.shape[1] == 1:
                # Squeeze unnecessary dim if it's [batch_size, 1]
                self.labels = self.labels.squeeze(1)
            else:
                raise ValueError(f"Expected labels to be 1D tensor but got shape {self.labels.shape}. "
                                "Labels should have shape [batch_size] or [batch_size, 1].")
        
        # Calculate sequence lengths by counting non-padding tokens
        self.lengths = torch.sum(self.sequences != pad_idx, dim=1).long()
        
        # Sanity check: ensure no zero-length sequences
        zero_lengths = (self.lengths == 0).sum().item()
        if zero_lengths > 0:
            print(f"Warning: Found {zero_lengths} sequences with zero length. "
                  f"First few lengths: {self.lengths[:10]}")
            # Optionally: self.lengths = torch.clamp(self.lengths, min=1)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32, pad_idx=0):
    """Create DataLoaders for train, validation and test sets."""
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, pad_idx)
    val_dataset = SentimentDataset(X_val, y_val, pad_idx)
    
    # Handle empty test set case (for LR finder)
    if len(X_test) == 0:
        test_dataset = None
    else:
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
    
    if test_dataset is None:
        test_loader = None
    else:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
    
    return train_loader, val_loader, test_loader 