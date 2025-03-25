import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionLayer(nn.Module):
    """Attention layer for BiLSTM."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        
    def forward(self, lstm_output, mask=None):
        # lstm_output shape: (batch_size, seq_len, hidden_dim*2)
        attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        
        if mask is not None:
            # Ensure mask has the same sequence length as lstm_output
            if mask.size(1) != lstm_output.size(1):
                print(f"Resizing mask from {mask.size(1)} to {lstm_output.size(1)}")
                if mask.size(1) > lstm_output.size(1):
                    # Truncate mask to match lstm_output sequence length
                    mask = mask[:, :lstm_output.size(1)]
                else:
                    # Pad mask to match lstm_output sequence length
                    pad_size = lstm_output.size(1) - mask.size(1)
                    mask = torch.nn.functional.pad(mask, (0, pad_size), value=0)
            
            # Apply mask to attention weights
            attention_weights = attention_weights.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights to lstm output
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_dim*2)
        
        return context, attention_weights

class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with attention for sentiment classification."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5, pad_idx=0):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        
        # Initialize weights
        self._init_weights()
        
        # Store embedding matrix for loading pre-trained embeddings
        self.embedding_matrix = None
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Initialize attention weights
        nn.init.xavier_normal_(self.attention.attention.weight)
        nn.init.constant_(self.attention.attention.bias, 0)
        
        # Initialize output layer weights
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, text, text_lengths=None, mask=None):
        """Forward pass of the model.
        
        Args:
            text: Padded input sequences [batch_size, max_seq_length=26]
            text_lengths: True sequence lengths before padding [batch_size]
            mask: Attention mask (1 for real tokens, 0 for padding) [batch_size, max_seq_length]
        """
        # Handle case where we receive a batch directly from LRFinder
        if isinstance(text, tuple) and len(text) >= 1:
            print("Model received tuple input, extracting first element as text")
            # This is likely from the DataLoader via LRFinder
            batch = text
            text = batch[0]
            if len(batch) > 1:
                text_lengths = batch[1]
        
        # Handle the case where text might not be on the correct device
        if not isinstance(text, torch.Tensor):
            print(f"Warning: Text is not a tensor, but {type(text)}")
            text = torch.tensor(text, device=self.fc.weight.device)
        
        # Move tensor to the model's device if needed
        if text.device != self.fc.weight.device:
            text = text.to(self.fc.weight.device)
        
        # Make our model more robust to different calling patterns
        if text_lengths is None:
            # If lengths not provided, calculate from input by counting non-zero tokens
            text_lengths = torch.sum(text != 0, dim=1).to(text.device)
        
        # Ensure text_lengths is on the right device
        if text_lengths.device != text.device:
            text_lengths = text_lengths.to(text.device)
        
        if mask is None:
            # If mask not provided, create it from input (1 for real tokens, 0 for padding)
            mask = (text != 0).to(text.device)
        
        # Sort sequences by true length (not padded length) for pack_padded_sequence
        lengths_sorted, sort_idx = text_lengths.sort(descending=True)
        text_sorted = text[sort_idx]
        mask_sorted = mask[sort_idx]
        
        # Embedding layer
        embedded = self.dropout(self.embedding(text_sorted))  # [batch_size, max_seq_length, embedding_dim]
        
        # Pack sequences using true lengths (ignores padding for LSTM computation)
        try:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_sorted.cpu(), batch_first=True
            )
            
            # LSTM processes only the real tokens, not padding
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            
            # Unpack to get regular tensor, padded to max length in batch
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )  # [batch_size, max_seq_length, hidden_dim*2]
        except Exception as e:
            print(f"Error in LSTM processing: {e}")
            print(f"Lengths: {lengths_sorted}")
            print(f"Text shape: {text_sorted.shape}")
            # Fall back to standard LSTM without packing
            output, (hidden, cell) = self.lstm(embedded)
        
        # Restore original sequence order
        _, unsort_idx = sort_idx.sort()
        output = output[unsort_idx]
        mask = mask_sorted[unsort_idx]
        
        # Apply attention (mask ensures attention only looks at real tokens)
        context, attention_weights = self.attention(output, mask)
        
        # Final layers
        context = self.dropout(context)
        output = self.fc(context)  # Return logits without sigmoid
        
        return output.squeeze(), attention_weights
    
    def load_pretrained_embeddings(self, embedding_matrix):
        """Load pre-trained embeddings into the embedding layer."""
        self.embedding_matrix = embedding_matrix
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False  # Freeze embeddings
    
    def predict(self, dataloader, device):
        """Generate predictions for a dataloader."""
        self.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                text, lengths, _ = batch
                text = text.to(device)
                lengths = lengths.to(device)
                
                # Create mask for attention
                mask = (text != 0).to(device)
                
                # Forward pass
                predictions, _ = self(text, lengths, mask)
                predictions = (predictions > 0.5).float()
                all_preds.extend(predictions.cpu().numpy())
        
        return all_preds