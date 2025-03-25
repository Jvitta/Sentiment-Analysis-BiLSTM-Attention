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
    
    def forward(self, text, text_lengths, mask=None):
        # text shape: (batch_size, seq_len)
        # text_lengths shape: (batch_size,)
        # mask shape: (batch_size, seq_len)
        
        # Embedding
        embedded = self.dropout(self.embedding(text))  # (batch_size, seq_len, embedding_dim)
        
        # Pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )  # (batch_size, seq_len, hidden_dim*2)
        
        # Apply attention
        context, attention_weights = self.attention(output, mask)
        
        # Dropout
        context = self.dropout(context)
        
        # Output layer
        output = torch.sigmoid(self.fc(context))
        
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