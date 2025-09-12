import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

# Simple implementation for sentence-based P/NP Oscillator

def create_simple_sentence_data():
    """Create a very simple sentence dataset for testing."""
    # Simple vocabulary
    vocab = ['[PAD]', '[START]', '[END]', 'the', 'cat', 'dog', 'bird', 'runs', 'jumps', 'flies', 'in', 'park', 'home']
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}
    
    # Simple sentences
    sentences = [
        "the cat runs in the park",
        "the dog jumps in the park", 
        "the bird flies in the park",
        "the cat runs in the home",
        "the dog jumps in the home",
        "the bird flies in the home"
    ]
    
    # Convert to sequences for next-word prediction
    inputs = []
    targets = []
    
    for sentence in sentences:
        words = sentence.split()
        # Add start token
        tokens = [word_to_idx['[START]']] + [word_to_idx.get(word, word_to_idx['[PAD]']) for word in words] + [word_to_idx['[END]']]
        
        # Create input-target pairs
        for i in range(len(tokens) - 1):
            # Input sequence up to position i
            input_seq = tokens[:i+1]
            # Pad to fixed length
            input_seq = input_seq + [word_to_idx['[PAD]']] * (10 - len(input_seq))
            # Target is next token
            target = tokens[i+1]
            
            inputs.append(input_seq)
            targets.append(target)
    
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long), word_to_idx, idx_to_word

class SimpleSentenceTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = self._create_pos_encoding(10, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.compression_events = []
        self.phase = 0.0
        
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        x = self.transformer(x)
        # Use last token for prediction
        x = x[:, -1, :]
        logits = self.fc_out(x)
        return logits

def train_simple_sentence_model():
    # Create data
    X, y, word_to_idx, idx_to_word = create_simple_sentence_data()
    vocab_size = len(word_to_idx)
    
    print(f"Dataset: {X.shape[0]} samples")
    print(f"Vocabulary: {vocab_size} words")
    print(f"Sample input: {X[0]}")
    print(f"Sample target: {y[0]} ({idx_to_word[y[0].item()]})")
    
    # Create model
    model = SimpleSentenceTransformer(vocab_size, d_model=32, nhead=2, num_layers=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X, y = X.to(device), y.to(device)
    
    print("Training...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Test
    model.eval()
    test_inputs = [
        [word_to_idx['[START]'], word_to_idx['the'], word_to_idx['cat'], word_to_idx['runs'], 0, 0, 0, 0, 0, 0],
        [word_to_idx['[START]'], word_to_idx['the'], word_to_idx['dog'], 0, 0, 0, 0, 0, 0, 0]
    ]
    
    print("\nTesting:")
    with torch.no_grad():
        for test_input in test_inputs:
            input_tensor = torch.tensor([test_input], dtype=torch.long).to(device)
            logits = model(input_tensor)
            pred = logits.argmax(dim=-1).item()
            print(f"Input: {[idx_to_word.get(i, '[UNK]') for i in test_input if i != 0]}")
            print(f"Prediction: {idx_to_word.get(pred, '[UNK]')}")
    
    return model

if __name__ == "__main__":
    model = train_simple_sentence_model()