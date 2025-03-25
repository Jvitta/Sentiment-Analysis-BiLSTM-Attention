import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis."""
    def __init__(self, sequences, labels, pad_idx=0):
        self.sequences = sequences
        self.labels = labels
        self.pad_idx = pad_idx
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
    
    @staticmethod
    def collate_fn(batch, pad_idx=0):
        """Collate function for DataLoader."""
        sequences, labels = zip(*batch)
        
        # Convert sequences to tensors and pad
        sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
        
        # Create sequence lengths tensor
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.float)
        
        return padded_sequences, lengths, labels


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32, pad_idx=0):
    """Create DataLoaders for train, validation and test sets."""
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, pad_idx)
    val_dataset = SentimentDataset(X_val, y_val, pad_idx)
    test_dataset = SentimentDataset(X_test, y_test, pad_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda b: SentimentDataset.collate_fn(b, pad_idx)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda b: SentimentDataset.collate_fn(b, pad_idx)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda b: SentimentDataset.collate_fn(b, pad_idx)
    )
    
    return train_loader, val_loader, test_loader 