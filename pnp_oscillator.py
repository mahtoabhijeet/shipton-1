import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using {DEVICE}")

# Create vocab: A, B, C, D, E, PAD, START
vocab = ["A", "B", "C", "D", "E", "<PAD>", "<START>"]
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
vocab_size = len(vocab)

# Patterns: "A B C" → "D", "B C D" → "E", etc.
patterns = [
    (["A", "B", "C"], "D"),
    (["B", "C", "D"], "E"),
    (["C", "D", "E"], "A"),  # wrap around
    (["D", "E", "A"], "B"),
    (["E", "A", "B"], "C"),
]

# Convert to tensors
def encode(seq):
    return torch.tensor([char_to_idx[ch] for ch in seq], dtype=torch.long)

def decode(tensor):
    return [idx_to_char[i.item()] for i in tensor]

X = torch.stack([encode(x) for x, _ in patterns])  # [5, 3]
y = torch.tensor([char_to_idx[y] for _, y in patterns])  # [5]

X = X.to(DEVICE)
y = y.to(DEVICE)

print("✅ Dataset ready:", X.shape, y.shape)

class TinyPNPTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(10, d_model)  # Max 10 tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
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
        threshold = S.max() * 0.01  # Much looser threshold for tiny model
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

model = TinyPNPTransformer(vocab_size).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

phase_osc = lambda step: (1 + math.sin(2 * math.pi * step / 20)) / 2  # Period = 20 steps

print("🧠 Training Tiny P/NP Oscillator...")

losses = []
events = []

for step in range(200):  # Only 200 steps needed
    model.train()
    model.phase = phase_osc(step)
    model.global_step = step
    
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    events.extend(model.compression_events)
    model.compression_events = []
    
    if step % 20 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f} | Phase: {model.phase:.2f} | Events: {len(events)}")

print("✅ Training complete.")

model.eval()
test_input = encode(["A", "B", "C"]).unsqueeze(0).to(DEVICE)  # [1, 3]
with torch.no_grad():
    logits = model(test_input)
    pred = logits.argmax(dim=-1).item()
    pred_char = idx_to_char[pred]

result_text = f"\n🧪 Test: 'A B C' → '{pred_char}' (Expected: 'D')"

# Show compression events
if events:
    result_text += f"\n\n🔍 COMPRESSION EVENTS DETECTED:"
    for e in events[:5]:  # Show first 5
        result_text += f"\n  Step {e['step']}: Rank {e['rank_before']} → {e['rank_after']} (Phase {e['phase']:.2f})"
else:
    result_text += "\n\n⚠️ No compression events detected — try loosening rank threshold."

# Save results to markdown file
with open("/Users/abhijeetmahto/shipton-1/results.md", "w") as f:
    f.write("# P/NP Oscillating Transformer Results\n\n")
    f.write("## Training Progress\n\n")
    for step in range(0, 200, 20):
        if step < len(losses):
            f.write(f"- Step {step} | Loss: {losses[step]:.4f} | Phase: {phase_osc(step):.2f} | Events: {len([e for e in events if e['step'] <= step])}\n")
    
    f.write(result_text)
    
    f.write("\n\n## Model Details\n\n")
    f.write(f"- Device: {DEVICE}\n")
    f.write(f"- Dataset shape: {X.shape}, {y.shape}\n")
    f.write(f"- Total compression events detected: {len(events)}\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    f.write(f"- Total parameters: {total_params}\n")

print(result_text)
print("\n✅ Results saved to results.md")