import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from sentence_dataset import create_sentence_dataset
from sentence_model import SentencePNPTransformer

def train_sentence_pnp_oscillator(
    num_sentences=200,
    batch_size=16,
    epochs=100,
    d_model=128,
    nhead=4,
    num_layers=2,
    learning_rate=0.001,
    max_seq_length=20
):
    # Create dataset
    print("Creating sentence dataset...")
    inputs, targets, tokenizer, sentences = create_sentence_dataset(num_sentences, max_seq_length)
    
    # Convert to tensors
    X = torch.tensor(inputs, dtype=torch.long)
    # For next word prediction, target should be the next word after the input sequence
    # Let's use the last meaningful token as target
    y = torch.tensor([seq[min(len(seq)-1, max_seq_length-1)] for seq in targets], dtype=torch.long)
    
    print(f"Dataset created: {X.shape}, {y.shape}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Check for any invalid values
    print(f"Input range: {X.min().item()} to {X.max().item()}")
    print(f"Target range: {y.min().item()} to {y.max().item()}")
    
    # Initialize model
    model = SentencePNPTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_length=max_seq_length
    )
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Gradient clipping to prevent exploding gradients
    gradient_clip = 1.0
    
    # P/NP phase oscillation
    phase_osc = lambda step: (1 + math.sin(2 * math.pi * step / 30)) / 2  # Period = 30 steps
    
    print("Training Sentence P/NP Oscillator...")
    
    losses = []
    events = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop
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
            
            # Check for NaN in logits
            if torch.isnan(logits).any():
                print(f"NaN detected in logits at epoch {epoch}, step {step}")
                continue
                
            loss = criterion(logits, batch_y)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch}, step {step}")
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            events.extend(model.compression_events)
            model.compression_events = []
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
        else:
            avg_loss = float('nan')
            losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Phase: {model.phase:.2f} | Events: {len(events)}")
    
    print("Training complete.")
    
    # Evaluation
    model.eval()
    test_sentences = [
        "the cat",
        "the dog runs",
        "the bird flies in the park"
    ]
    
    print("\nTesting sentence completion:")
    with torch.no_grad():
        for sentence in test_sentences:
            # Encode input
            input_seq = tokenizer.encode(sentence, max_seq_length)
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
            
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
    
    # Save results
    results = {
        'model_params': sum(p.numel() for p in model.parameters()),
        'vocab_size': tokenizer.vocab_size,
        'dataset_size': len(inputs),
        'final_loss': losses[-1] if losses and not math.isnan(losses[-1]) else 0,
        'compression_events': len(events),
        'tokenizer': tokenizer
    }
    
    return model, results, losses, events

# Example usage
if __name__ == "__main__":
    model, results, losses, events = train_sentence_pnp_oscillator(
        num_sentences=100,
        batch_size=8,
        epochs=50,
        d_model=64,  # Smaller model for stability
        nhead=4,
        num_layers=1,  # Fewer layers for stability
        learning_rate=0.001
    )
    
    print("\nResults Summary:")
    print(f"Model parameters: {results['model_params']:,}")
    print(f"Vocabulary size: {results['vocab_size']}")
    print(f"Dataset size: {results['dataset_size']}")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Compression events: {results['compression_events']}")