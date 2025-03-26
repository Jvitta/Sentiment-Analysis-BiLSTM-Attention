import numpy as np
import torch
from data_loader import SentimentDataset, create_dataloaders

def test_sentiment_dataset():
    """Test the SentimentDataset class with various input shapes."""
    print("=== TESTING SENTIMENTDATASET CLASS ===")
    
    # Test 1: Load real data files
    print("\nTest 1: Loading real data from NumPy files")
    try:
        X_train = np.load('data/processed_data/X_train.npy')
        y_train = np.load('data/processed_data/y_train.npy')
        
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        # Create dataset
        train_dataset = SentimentDataset(X_train, y_train)
        print(f"Dataset created successfully: {len(train_dataset)} samples")
        
        # Test first batch
        sequences, lengths, labels = train_dataset[0]
        print(f"First sample shapes - sequence: {sequences.shape}, length: {lengths.shape}, label: {labels.shape}")
        print("Test 1: PASSED ✓")
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
    
    # Test 2: Test with empty test set (for LR finder)
    print("\nTest 2: Empty test set handling")
    try:
        X_train = np.random.randint(0, 100, size=(10, 5))
        y_train = np.random.randint(0, 2, size=10)
        X_val = np.random.randint(0, 100, size=(5, 5))
        y_val = np.random.randint(0, 2, size=5)
        X_test = []
        y_test = []
        
        train_loader, val_loader, test_loader = create_dataloaders(
            X_train, X_val, X_test, y_train, y_val, y_test, batch_size=2
        )
        
        print(f"train_loader created: {len(train_loader)} batches")
        print(f"val_loader created: {len(val_loader)} batches")
        print(f"test_loader is None: {test_loader is None}")
        print("Test 2: PASSED ✓")
    except Exception as e:
        print(f"Test 2 FAILED: {e}")
    
    # Test 3: Test with 2D labels (should be squeezed)
    print("\nTest 3: 2D labels handling")
    try:
        X = np.random.randint(0, 100, size=(10, 5))
        y = np.random.randint(0, 2, size=(10, 1))  # 2D labels
        
        dataset = SentimentDataset(X, y)
        # Check first item
        _, _, label = dataset[0]
        print(f"Label shape after squeezing: {label.shape}")
        print(f"Label is scalar: {label.ndim == 0}")
        print("Test 3: PASSED ✓")
    except Exception as e:
        print(f"Test 3 FAILED: {e}")
    
    print("\n=== ALL TESTS COMPLETED ===")

if __name__ == "__main__":
    test_sentiment_dataset() 