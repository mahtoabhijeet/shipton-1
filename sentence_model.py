import torch
import torch.nn as nn
import math

class SentencePNPTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_seq_length=50):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = self._create_pos_encoding(max_seq_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=512, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # P/NP Control
        self.phase = 0.0  # 0 = P (compress), 1 = NP (explore)
        self.compression_events = []
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def estimate_rank(self, tensor):
        if tensor.ndim != 2: return 0
        # Use a more robust SVD implementation for sentence data
        tensor_f32 = tensor.to(torch.float32)
        try:
            U, S, Vh = torch.linalg.svd(tensor_f32, full_matrices=False)
            # Adjust threshold for sentence-level data
            threshold = S.max() * 0.05  # More appropriate for sentence data
            return (S > threshold).sum().item()
        except:
            return 0

    def forward(self, x):
        # x shape: [B, T] where B is batch size, T is sequence length
        x = self.embedding(x)  # [B, T, D]
        
        # Ensure positional encoding matches sequence length
        seq_len = x.size(1)
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        
        # Track stats BEFORE transformer (simulate layer 0)
        rank_before = None
        if self.training and hasattr(self, 'phase') and self.phase < 0.5:
            # Sample a few tokens for rank estimation (more efficient for longer sequences)
            sample = x[0, :min(5, seq_len), :].detach()  # First 5 tokens or fewer
            rank_before = self.estimate_rank(sample)
        
        x = self.transformer(x)  # [B, T, D]
        
        # For sentence completion, we typically predict the next token
        # So we use the last token's representation
        x = x[:, -1, :]  # [B, D] - last token representation
        logits = self.fc_out(x)  # [B, vocab_size]
        
        # Compression event: if rank dropped significantly after transformer
        if (self.training and hasattr(self, 'phase') and self.phase < 0.5 and 
            rank_before is not None):
            sample_after = x[0].detach().unsqueeze(0)  # [1, D]
            rank_after = self.estimate_rank(sample_after)
            # More sensitive detection for sentence data
            if rank_before - rank_after > 0.5:  # Lower threshold for sentence data
                self.compression_events.append({
                    'step': getattr(self, 'global_step', 0),
                    'rank_before': rank_before,
                    'rank_after': rank_after,
                    'phase': self.phase
                })
        
        return logits

# Alternative model for sequence-to-sequence prediction
class SentenceSeq2SeqPNPTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_seq_length=50):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = self._create_pos_encoding(max_seq_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=512, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # P/NP Control
        self.phase = 0.0  # 0 = P (compress), 1 = NP (explore)
        self.compression_events = []
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def estimate_rank(self, tensor):
        if tensor.ndim != 2: return 0
        # Use a more robust SVD implementation for sentence data
        tensor_f32 = tensor.to(torch.float32)
        try:
            U, S, Vh = torch.linalg.svd(tensor_f32, full_matrices=False)
            # Adjust threshold for sentence-level data
            threshold = S.max() * 0.05  # More appropriate for sentence data
            return (S > threshold).sum().item()
        except:
            return 0

    def forward(self, x):
        # x shape: [B, T] where B is batch size, T is sequence length
        x = self.embedding(x)  # [B, T, D]
        
        # Ensure positional encoding matches sequence length
        seq_len = x.size(1)
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        
        # Track stats BEFORE transformer (simulate layer 0)
        rank_before = None
        if self.training and hasattr(self, 'phase') and self.phase < 0.5:
            # Sample a few tokens for rank estimation
            sample = x[0, :min(5, seq_len), :].detach()  # First 5 tokens or fewer
            rank_before = self.estimate_rank(sample)
        
        x = self.transformer(x)  # [B, T, D]
        
        # For sequence-to-sequence, we predict for each position
        logits = self.fc_out(x)  # [B, T, vocab_size]
        
        # Compression event: if rank dropped significantly after transformer
        if (self.training and hasattr(self, 'phase') and self.phase < 0.5 and 
            rank_before is not None):
            # Sample from the output for rank estimation
            sample_after = x[0, :min(5, seq_len), :].detach()  # First 5 tokens or fewer
            rank_after = self.estimate_rank(sample_after)
            # More sensitive detection for sentence data
            if rank_before - rank_after > 0.5:  # Lower threshold for sentence data
                self.compression_events.append({
                    'step': getattr(self, 'global_step', 0),
                    'rank_before': rank_before,
                    'rank_after': rank_after,
                    'phase': self.phase
                })
        
        return logits

# Example usage
if __name__ == "__main__":
    # Test the model with sample data
    vocab_size = 100
    model = SentencePNPTransformer(vocab_size, d_model=128, nhead=4, num_layers=2)
    
    # Sample input (batch_size=2, seq_len=10)
    sample_input = torch.randint(1, vocab_size, (2, 10))
    
    # Test forward pass
    with torch.no_grad():
        output = model(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")