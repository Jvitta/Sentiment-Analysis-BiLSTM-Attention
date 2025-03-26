import numpy as np
import os

def examine_data_file(file_path):
    """
    Examine the shape and content of a NumPy file.
    
    Args:
        file_path: Path to the numpy array file
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Load the data
    data = np.load(file_path)
    
    # Print basic information
    print(f"\nFile: {file_path}")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    # Check dimensions
    if len(data.shape) == 1:
        print("WARNING: This is a 1D array, not a 2D array of sequences!")
        print(f"First few elements: {data[:5]}")
        return
    
    # Count non-padding tokens in each sequence (where padding token is 0)
    non_padding_counts = np.sum(data != 0, axis=1)
    
    # Find sequences with zero non-padding tokens
    zero_indices = np.where(non_padding_counts == 0)[0]
    has_zero_lengths = len(zero_indices) > 0
    
    if has_zero_lengths:
        print(f"WARNING: Found {len(zero_indices)} sequences with zero length")
        print(f"Zero-length indices: {zero_indices}")
    else:
        print(f"No zero-length sequences found (total: {len(data)})")
    
    # Print length statistics
    print(f"Min sequence length (non-zero tokens): {np.min(non_padding_counts)}")
    print(f"Max sequence length (non-zero tokens): {np.max(non_padding_counts)}")
    print(f"Mean sequence length (non-zero tokens): {np.mean(non_padding_counts):.2f}")
    
    # Display a few examples
    print("\nFirst 3 sequences:")
    for i in range(min(3, len(data))):
        seq = data[i]
        non_zeros = np.count_nonzero(seq)
        print(f"  Sequence {i}: {non_zeros} non-zero tokens, sequence: {seq}")

def main():
    """Examine the data files."""
    # Define data directories
    data_dir = 'data/processed_data'
    
    # List of files to examine
    files_to_check = [
        'X_train.npy', 
        'X_val.npy', 
        'X_test.npy',
        'y_train.npy',
        'y_val.npy',
        'y_test.npy'
    ]
    
    print("=== DATA EXAMINATION ===")
    
    # Examine each file
    for filename in files_to_check:
        file_path = os.path.join(data_dir, filename)
        examine_data_file(file_path)
    
    print("\n=== FINISHED EXAMINATION ===")

if __name__ == "__main__":
    main()
