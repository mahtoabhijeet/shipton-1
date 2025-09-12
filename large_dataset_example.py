import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using {DEVICE}")

# Extended vocabulary with more characters
vocab = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "<PAD>", "<START>"]
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
vocab_size = len(vocab)

# Generate larger dataset with more complex patterns
def generate_patterns(num_patterns=100, seq_length=5):
    patterns = []
    for _ in range(num_patterns):
        # Generate random sequence
        seq = random.sample(vocab[:-2], seq_length)  # Exclude <PAD> and <START>
        # Determine next character (simple rule: next in alphabetical order, wrapping around)
        last_char = seq[-1]
        last_idx = char_to_idx[last_char]
        next_idx = (last_idx + 1) % (vocab_size - 2)  # -2 to exclude special tokens
        next_char = idx_to_char[next_idx]
        patterns.append((seq, next_char))
    return patterns

# Generate larger dataset
large_patterns = generate_patterns(100, 5)  # 100 patterns of 5-character sequences

# Convert to tensors
def encode(seq):
    return torch.tensor([char_to_idx[ch] for ch in seq], dtype=torch.long)

def decode(tensor):
    return [idx_to_char[i.item()] for i in tensor]

# Create batches
batch_size = 10
num_batches = len(large_patterns) // batch_size

# Prepare all data
X_all = torch.stack([encode(x) for x, _ in large_patterns])
y_all = torch.tensor([char_to_idx[y] for _, y in large_patterns])

print("✅ Large Dataset ready:", X_all.shape, y_all.shape)

class TinyPNPTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(20, d_model)  # Max 20 tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # P/NP Control
        self.phase = 0.0  # 0 = P (compress), 1 = NP (explore)
        self.compression_events = []

    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def estimate_rank(self, tensor):
        if tensor.ndim != 2: return 0
        tensor_f32 = tensor.to(torch.float32)
        U, S, Vh = torch.linalg.svd(tensor_f32, full_matrices=False)
        threshold = S.max() * 0.01  # Looser threshold for model
        return (S > threshold).sum().item()

    def forward(self, x):
        x = self.embedding(x)  # [B, T, D]
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        
        # Track stats BEFORE transformer (simulate layer 0)
        if self.training and hasattr(self, 'phase'):
            sample = x[0].detach()
            rank_before = self.estimate_rank(sample)
        
        x = self.transformer(x)  # [B, T, D]
        x = x.mean(dim=1)  # Global average pooling
        logits = self.fc_out(x)  # [B, vocab_size]
        
        # Compression event: if rank dropped significantly after transformer
        if self.training and hasattr(self, 'phase') and self.phase < 0.5:
            sample_after = x[0].detach()
            rank_after = self.estimate_rank(sample_after)
            # More sensitive detection
            if rank_before - rank_after > 1:  # Lower threshold for detection
                self.compression_events.append({
                    'step': getattr(self, 'global_step', 0),
                    'rank_before': rank_before,
                    'rank_after': rank_after,
                    'phase': self.phase
                })
        
        return logits

# Initialize model
model = TinyPNPTransformer(vocab_size).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower learning rate for larger dataset
criterion = nn.CrossEntropyLoss()

phase_osc = lambda step: (1 + math.sin(2 * math.pi * step / 50)) / 2  # Longer period for larger dataset

print("🧠 Training Tiny P/NP Oscillator with larger dataset...")

losses = []
events = []

# Training loop with batching
for epoch in range(50):  # More epochs for larger dataset
    epoch_loss = 0.0
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size
        
        X_batch = X_all[start_idx:end_idx].to(DEVICE)
        y_batch = y_all[start_idx:end_idx].to(DEVICE)
        
        step = epoch * num_batches + batch
        model.train()
        model.phase = phase_osc(step)
        model.global_step = step
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        events.extend(model.compression_events)
        model.compression_events = []
    
    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Phase: {model.phase:.2f} | Events: {len(events)}")

print("✅ Training complete with larger dataset.")

# Test with a few examples
model.eval()
test_samples = [
    ["A", "B", "C", "D", "E"],
    ["F", "G", "H", "I", "J"],
    ["K", "L", "M", "N", "O"]
]

print("\n🧪 Testing with larger dataset patterns:")
for sample in test_samples:
    test_input = encode(sample).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(test_input)
        pred = logits.argmax(dim=-1).item()
        pred_char = idx_to_char[pred]
        expected_idx = (char_to_idx[sample[-1]] + 1) % (vocab_size - 2)
        expected_char = idx_to_char[expected_idx]
        print(f"Test: {' '.join(sample)} → '{pred_char}' (Expected: '{expected_char}')")

# Show compression events
if events:
    print(f"\n🔍 COMPRESSION EVENTS DETECTED:")
    for e in events[:5]:  # Show first 5
        print(f"  Step {e['step']}: Rank {e['rank_before']} → {e['rank_after']} (Phase {e['phase']:.2f})")
else:
    print("\n⚠️ No compression events detected — try adjusting thresholds.")

# Save results
with open("/Users/abhijeetmahto/shipton-1/large_dataset_results.md", "w") as f:
    f.write("# P/NP Oscillator Results with Large Dataset\n\n")
    f.write("## Training Progress\n\n")
    for i in range(0, len(losses), 5):
        if i < len(losses):
            f.write(f"- Epoch {i} | Avg Loss: {losses[i]:.4f} | Phase: {phase_osc(i * num_batches):.2f} | Events: {len([e for e in events if e['step'] <= i * num_batches])}\n")
    
    f.write("\n## Test Results\n\n")
    for sample in test_samples:
        test_input = encode(sample).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(test_input)
            pred = logits.argmax(dim=-1).item()
            pred_char = idx_to_char[pred]
            expected_idx = (char_to_idx[sample[-1]] + 1) % (vocab_size - 2)
            expected_char = idx_to_char[expected_idx]
            f.write(f"Test: {' '.join(sample)} → '{pred_char}' (Expected: '{expected_char}')\n")
    
    f.write(f"\n## Model Details\n\n")
    f.write(f"- Device: {DEVICE}\n")
    f.write(f"- Dataset shape: {X_all.shape}, {y_all.shape}\n")
    f.write(f"- Total compression events detected: {len(events)}\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    f.write(f"- Total parameters: {total_params}\n")
    
    if events:
        f.write("\n## Sample Compression Events\n\n")
        for e in events[:5]:  # Show first 5
            f.write(f"- Step {e['step']}: Rank {e['rank_before']} → {e['rank_after']} (Phase {e['phase']:.2f})\n")

print("\n✅ Results with large dataset saved to large_dataset_results.md")