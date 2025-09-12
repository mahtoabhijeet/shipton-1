import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
from collections import Counter
import re

class SentenceTokenizer:
    def __init__(self, max_vocab_size=200):
        self.max_vocab_size = max_vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, sentences):
        # Count word frequencies
        word_counts = Counter()
        for sentence in sentences:
            # Simple tokenization - split by whitespace and clean punctuation
            words = re.findall(r'\b\w+\b', sentence.lower())
            word_counts.update(words)
        
        # Sort by frequency and limit vocabulary size
        sorted_words = word_counts.most_common(self.max_vocab_size-4)  # Reserve space for special tokens
        
        # Add special tokens
        self.word_to_idx = {'[PAD]': 0, 'the': 1, '[START]': 2, '[END]': 3, '[UNK]': 4}
        self.idx_to_word = {0: '[PAD]', 1: 'the', 2: '[START]', 3: '[END]', 4: '[UNK]'}
        
        # Add regular words
        for i, (word, _) in enumerate(sorted_words):
            if word not in self.word_to_idx:  # Avoid duplicates
                self.word_to_idx[word] = i + 5
                self.idx_to_word[i + 5] = word
            
        self.vocab_size = len(self.word_to_idx)
        
    def encode(self, sentence, max_length=20):
        # Simple tokenization - split by whitespace and clean punctuation
        words = re.findall(r'\b\w+\b', sentence.lower())
        
        # Start with [START] token
        indices = [self.word_to_idx.get('[START]', 2)]
        
        # Add words (or [UNK] for out-of-vocabulary)
        for word in words:
            idx = self.word_to_idx.get(word, self.word_to_idx.get('[UNK]', 4))
            indices.append(idx)
            
        # Add [END] token
        indices.append(self.word_to_idx.get('[END]', 3))
        
        # Pad or truncate to max_length
        if len(indices) < max_length:
            indices.extend([self.word_to_idx.get('[PAD]', 0)] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
            
        return indices
        
    def decode(self, indices):
        words = []
        for idx in indices:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word not in ['[PAD]', '[START]']:
                    if word == '[END]':
                        break
                    words.append(word)
        return ' '.join(words)

def generate_sentence_dataset(num_sentences=100):
    """
    Generate a dataset of simple sentences with predictable patterns.
    """
    subjects = ["the cat", "the dog", "the bird", "the fish", "the rabbit", 
                "john", "mary", "the child", "the teacher", "the student",
                "a man", "a woman", "the boy", "the girl", "the person"]
    
    verbs = ["runs", "jumps", "sleeps", "eats", "plays", "swims", "flies", 
             "reads", "writes", "sings", "dances", "walks", "talks", "laughs"]
    
    objects = ["in the park", "on the mat", "under the tree", "in the house", 
               "at school", "with a ball", "in the water", "on the roof", 
               "with a book", "very well", "quickly", "slowly", "loudly",
               "in the garden", "at home", "with friends"]
    
    adjectives = ["big", "small", "red", "blue", "happy", "sad", "fast", 
                  "slow", "loud", "quiet", "young", "old", "smart", "funny"]
    
    sentences = []
    
    # Pattern 1: Subject + Verb
    for _ in range(num_sentences // 4):
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        sentences.append(f"{subject} {verb}")
    
    # Pattern 2: Subject + Verb + Object
    for _ in range(num_sentences // 4):
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        sentences.append(f"{subject} {verb} {obj}")
    
    # Pattern 3: Subject + Adjective + Verb
    for _ in range(num_sentences // 4):
        subject = random.choice(subjects)
        adj = random.choice(adjectives)
        verb = random.choice(verbs)
        sentences.append(f"{subject} is {adj} and {verb}")
    
    # Pattern 4: Adjective + Subject + Verb + Object
    for _ in range(num_sentences // 4):
        adj = random.choice(adjectives)
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        sentences.append(f"{adj} {subject} {verb} {obj}")
    
    return sentences

def create_sentence_sequences(sentences, tokenizer, max_length=15):
    """
    Create input-target pairs for next-word prediction.
    """
    inputs = []
    targets = []
    
    for sentence in sentences:
        # Encode the full sentence
        full_encoding = tokenizer.encode(sentence, max_length)
        
        # For each position, create an input-target pair
        for i in range(1, len(full_encoding) - 1):
            # Input is from start to current position
            input_seq = full_encoding[:i] + [0] * (max_length - i)
            # Target is the next token
            target_token = full_encoding[i]
            
            # Only add if input has meaningful content
            if any(idx != 0 for idx in input_seq):
                inputs.append(input_seq)
                targets.append(target_token)
    
    return inputs, targets

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
            if p.dim() > 1:  # Fixed: p.dim() instead of p.dim
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

def train_sentence_model():
    # Generate dataset
    print("Generating sentence dataset...")
    sentences = generate_sentence_dataset(50)
    
    # Create tokenizer and build vocabulary
    tokenizer = SentenceTokenizer(max_vocab_size=100)
    tokenizer.build_vocab(sentences)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create sequences for training
    inputs, targets = create_sentence_sequences(sentences, tokenizer, max_length=15)
    
    # Convert to tensors
    X = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    
    print(f"Dataset size: {len(inputs)} input-target pairs")
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    
    # Initialize model
    model = SentencePNPTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_layers=1,
        max_seq_length=15
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # P/NP phase oscillation
    phase_osc = lambda step: (1 + math.sin(2 * math.pi * step / 20)) / 2  # Period = 20 steps
    
    print("Training Sentence P/NP Oscillator...")
    
    losses = []
    events = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop
    batch_size = 16
    epochs = 50
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process in batches
        for i in range(0, len(X_shuffled), batch_size):
            batch_X = X_shuffled[i:i+batch_size].to(device)
            batch_y = y_shuffled[i:i+batch_size].to(device)
            
            step = epoch * (len(X_shuffled) // batch_size) + (i // batch_size)
            
            model.train()
            model.phase = phase_osc(step)
            model.global_step = step
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            events.extend(model.compression_events)
            model.compression_events = []
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Phase: {model.phase:.2f} | Events: {len(events)}")
    
    print("Training complete.")
    
    # Evaluation
    model.eval()
    test_sentences = [
        "the cat",
        "the dog runs",
        "the bird is happy and"
    ]
    
    print("\nTesting sentence completion:")
    with torch.no_grad():
        for sentence in test_sentences:
            # Encode input
            input_seq = tokenizer.encode(sentence, 15)
            # Find the last non-pad token
            seq_len = 0
            for i in range(len(input_seq)):
                if input_seq[i] != 0:
                    seq_len = i + 1
                else:
                    break
            
            # Trim to actual sequence length
            input_seq_trimmed = input_seq[:seq_len] if seq_len > 0 else [2]  # At least [START]
            input_tensor = torch.tensor([input_seq_trimmed + [0] * (15 - len(input_seq_trimmed))], dtype=torch.long).to(device)
            
            # Get prediction
            logits = model(input_tensor)
            pred_idx = logits.argmax(dim=-1).item()
            
            # Decode prediction
            if pred_idx in tokenizer.idx_to_word:
                pred_word = tokenizer.idx_to_word[pred_idx]
                print(f"Input: '{sentence}' → Prediction: '{pred_word}'")
            else:
                print(f"Input: '{sentence}' → Prediction: [UNK_TOKEN_{pred_idx}]")
    
    # Show compression events
    if events:
        print(f"\nCompression events detected: {len(events)}")
        print("Sample events:")
        for e in events[:5]:  # Show first 5
            print(f"  Step {e['step']}: Rank {e['rank_before']:.1f} → {e['rank_after']:.1f} (Phase {e['phase']:.2f})")
    else:
        print("\nNo compression events detected.")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = train_sentence_model()