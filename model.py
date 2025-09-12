import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from collections import Counter
import re
import time

class MemoryEfficientTransformer(nn.Module):
    """Memory-efficient transformer with P/NP oscillation and compression detection."""
    
    def __init__(self, vocab_size, d_model=384, nhead=8, num_layers=4, max_seq_length=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = self._create_pos_encoding(max_seq_length, d_model)
        
        # Transformer encoder with memory optimizations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=1024, dropout=dropout, 
            batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # P/NP control
        self.phase = 0.0  # 0 = P (compress), 1 = NP (explore)
        self.compression_events = []
        self.global_step = 0
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _create_pos_encoding(self, max_len, d_model):
        """Create positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def estimate_rank(self, tensor):
        """Estimate matrix rank using SVD."""
        if tensor.ndim != 2 or tensor.numel() == 0:
            return 0
            
        try:
            # Use CPU for SVD operations (MPS doesn't support SVD well)
            tensor_cpu = tensor.cpu().to(torch.float32)
            
            # Compute SVD
            U, S, Vh = torch.linalg.svd(tensor_cpu, full_matrices=False)
            
            # Estimate rank with adaptive threshold
            threshold = S.max() * 0.01  # More sensitive for language tasks
            rank = (S > threshold).sum().item()
            
            return rank
        except Exception as e:
            # Fallback in case of numerical issues
            return min(tensor.shape)

    def forward(self, x):
        """Forward pass with compression event detection."""
        batch_size, seq_len = x.shape
        
        # Embedding and positional encoding
        x = self.embedding(x)  # [B, T, D]
        
        # Add positional encoding (ensure dimensions match)
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        
        # Track stats BEFORE transformer for compression detection
        rank_before = None
        if self.training and hasattr(self, 'phase') and self.phase < 0.5:
            # Sample a representative subset for efficiency
            sample_size = min(3, batch_size)
            token_sample = min(5, seq_len)
            sample = x[:sample_size, :token_sample, :].detach()
            rank_before = self.estimate_rank(sample.reshape(-1, self.d_model))
        
        # Transformer forward pass
        x = self.transformer(x)  # [B, T, D]
        
        # Use last token representation for prediction
        x = x[:, -1, :]  # [B, D]
        logits = self.fc_out(x)  # [B, vocab_size]
        
        # Compression event detection
        if (self.training and hasattr(self, 'phase') and self.phase < 0.5 and 
            rank_before is not None):
            # Estimate rank after transformation
            sample_after = x[:min(3, batch_size), :].detach()
            rank_after = self.estimate_rank(sample_after)
            
            # Detect compression event
            if rank_before - rank_after > 1.0:  # Adaptive threshold
                self.compression_events.append({
                    'step': getattr(self, 'global_step', 0),
                    'rank_before': rank_before,
                    'rank_after': rank_after,
                    'phase': self.phase,
                    'batch_size': batch_size
                })
        
        return logits

class SentenceTokenizer:
    """Simple tokenizer for sentence processing."""
    
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            # Simple tokenization
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts.update(words)
        
        # Sort by frequency and limit vocabulary
        sorted_words = word_counts.most_common(self.max_vocab_size - 4)
        
        # Add special tokens
        self.word_to_idx = {
            '[PAD]': 0,
            '[UNK]': 1, 
            '[START]': 2,
            '[END]': 3
        }
        self.idx_to_word = {
            0: '[PAD]',
            1: '[UNK]',
            2: '[START]',
            3: '[END]'
        }
        
        # Add regular words
        for i, (word, _) in enumerate(sorted_words):
            self.word_to_idx[word] = i + 4
            self.idx_to_word[i + 4] = word
            
        self.vocab_size = len(self.word_to_idx)
        
    def encode(self, text, max_length=100):
        """Encode text to token indices."""
        # Tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Convert to indices
        indices = [self.word_to_idx.get('[START]', 2)]
        for word in words:
            idx = self.word_to_idx.get(word, self.word_to_idx.get('[UNK]', 1))
            indices.append(idx)
        indices.append(self.word_to_idx.get('[END]', 3))
        
        # Pad or truncate
        if len(indices) < max_length:
            indices.extend([self.word_to_idx.get('[PAD]', 0)] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
            
        return indices
        
    def decode(self, indices):
        """Decode token indices to text."""
        words = []
        for idx in indices:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word not in ['[PAD]', '[START]', '[UNK]']:
                    if word == '[END]':
                        break
                    words.append(word)
        return ' '.join(words)

def create_sentence_dataset(texts, tokenizer, max_length=100):
    """Create training dataset from texts."""
    inputs = []
    targets = []
    
    for text in texts:
        # Encode full text
        full_encoding = tokenizer.encode(text, max_length)
        
        # Create input-target pairs
        for i in range(1, len(full_encoding) - 1):
            if full_encoding[i] != 0:  # Skip padding
                input_seq = full_encoding[:i] + [0] * (max_length - i)
                target_token = full_encoding[i]
                inputs.append(input_seq)
                targets.append(target_token)
    
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def train_model(model, train_loader, optimizer, criterion, device, epochs=10):
    """Train the model with P/NP oscillation."""
    model.to(device)
    losses = []
    events = []
    
    # P/NP phase oscillation (period of 30 steps)
    phase_osc = lambda step: (1 + math.sin(2 * math.pi * step / 30)) / 2
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Update phase
            step = epoch * len(train_loader) + batch_idx
            model.phase = phase_osc(step)
            model.global_step = step
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Collect compression events
            events.extend(model.compression_events)
            model.compression_events = []
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Print progress
        if epoch % 2 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | "
                  f"Phase: {model.phase:.2f} | Events: {len(events)} | "
                  f"Time: {elapsed:.1f}s")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    
    return losses, events

def evaluate_model(model, test_sentences, tokenizer, device, max_length=50):
    """Evaluate model on test sentences."""
    model.eval()
    print("\nSample predictions:")
    
    with torch.no_grad():
        for sentence in test_sentences:
            # Encode input
            input_seq = tokenizer.encode(sentence, max_length)
            
            # Find actual sequence length
            seq_len = 0
            for i in range(len(input_seq)):
                if input_seq[i] != 0:
                    seq_len = i + 1
                else:
                    break
            
            # Prepare input tensor (ensure correct length)
            input_seq_trimmed = input_seq[:seq_len] if seq_len > 0 else [2]
            # Pad to max_length
            input_seq_padded = input_seq_trimmed + [0] * (max_length - len(input_seq_trimmed))
            input_tensor = torch.tensor([input_seq_padded], dtype=torch.long).to(device)
            
            # Get prediction
            logits = model(input_tensor)
            pred_idx = logits.argmax(dim=-1).item()
            
            # Decode prediction
            if pred_idx in tokenizer.idx_to_word:
                pred_word = tokenizer.idx_to_word[pred_idx]
                print(f"Input: '{sentence}' → Prediction: '{pred_word}'")
            else:
                print(f"Input: '{sentence}' → Prediction: [UNK_TOKEN_{pred_idx}]")

# Model information function
def get_model_info(model):
    """Get model parameter count and memory usage."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Rough memory estimation (parameters + gradients)
    memory_estimate = (total_params * 4 * 2) / (1024 ** 2)  # MB
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_estimate_mb': memory_estimate
    }