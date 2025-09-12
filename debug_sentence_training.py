import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from sentence_dataset import create_sentence_dataset
from sentence_model import SentencePNPTransformer

# First, let's debug the dataset creation
def debug_dataset():
    print("Debugging dataset creation...")
    inputs, targets, tokenizer, sentences = create_sentence_dataset(20, 15)
    
    print("Sample sentences:")
    for i in range(3):
        print(f"{i+1}. {sentences[i]}")
    
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Sample word to index mappings:")
    for i, (word, idx) in enumerate(list(tokenizer.word_to_idx.items())[:10]):
        print(f"  {word}: {idx}")
    
    print(f"\nSample encodings:")
    for i in range(3):
        input_seq = inputs[i]
        target_seq = targets[i]
        decoded_input = tokenizer.decode(input_seq)
        print(f"Input sequence:  {input_seq}")
        print(f"Target sequence: {target_seq}")
        print(f"Decoded input:   '{decoded_input}'")
        print()

def create_proper_targets(inputs, targets):
    """
    Create proper targets for next-word prediction.
    For each input sequence, the target should be the next word.
    """
    # For sentence completion, we want to predict the next word
    # Let's create a proper target where target[i] is the next word after input[i]
    
    proper_targets = []
    for i in range(len(inputs)):
        input_seq = inputs[i]
        target_seq = targets[i]
        
        # Find the last non-padding token in input
        last_token_idx = 0
        for j in range(len(input_seq)-1, -1, -1):
            if input_seq[j] != 0:  # Not padding
                last_token_idx = j
                break
        
        # The target is the token that comes after the last input token
        # In our case, this would be target_seq[last_token_idx] if it exists
        if last_token_idx < len(target_seq):
            proper_targets.append(target_seq[last_token_idx])
        else:
            # Fallback to a common token
            proper_targets.append(4)  # Usually 'the' in our vocabulary
    
    return torch.tensor(proper_targets, dtype=torch.long)

def train_sentence_pnp_oscillator_debug(
    num_sentences=50,  # Smaller dataset for debugging
    batch_size=8,
    epochs=30,
    d_model=64,
    nhead=4,
    num_layers=1,
    learning_rate=0.001,
    max_seq_length=15
):
    # Create dataset
    print("Creating sentence dataset...")
    inputs, targets, tokenizer, sentences = create_sentence_dataset(num_sentences, max_seq_length)
    
    # Debug the dataset
    print(f"Dataset created: {len(inputs)} sentences")
    print(f"Sample inputs shape: {len(inputs[0])}")
    print(f"Sample targets shape: {len(targets[0])}")
    
    # Convert to tensors
    X = torch.tensor(inputs, dtype=torch.long)
    
    # Create proper targets
    y = create_proper_targets(inputs, targets)
    
    print(f"Input tensor shape: {X.shape}")
    print(f"Target tensor shape: {y.shape}")
    print(f"Target value range: {y.min().item()} to {y.max().item()}")
    
    # Check for any invalid values
    print(f"Input range: {X.min().item()} to {X.max().item()}")
    
    # Verify targets are within vocabulary range
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    invalid_targets = (y < 0) | (y >= vocab_size)
    if invalid_targets.any():
        print(f"WARNING: {invalid_targets.sum().item()} invalid target values found")
        # Fix invalid targets
        y = torch.clamp(y, 0, vocab_size-1)
    
    # Initialize model
    model = SentencePNPTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_length=max_seq_length
    )
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # P/NP phase oscillation
    phase_osc = lambda step: (1 + math.sin(2 * math.pi * step / 20)) / 2  # Period = 20 steps
    
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
                print(f"Logits range: {logits.min().item()} to {logits.max().item()}")
                continue
                
            loss = criterion(logits, batch_y)
            
            # Check for NaN or inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at epoch {epoch}, step {step}: {loss.item()}")
                print(f"Logits sample: {logits[0][:5].detach().cpu().numpy()}")
                print(f"Targets sample: {batch_y[:5].detach().cpu().numpy()}")
                continue
                
            loss.backward()
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
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Phase: {model.phase:.2f} | Events: {len(events)}")
    
    print("Training complete.")
    
    # Evaluation
    model.eval()
    test_sentences = [
        "the cat",
        "the dog runs",
        "the bird flies"
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

# Debug the dataset first
if __name__ == "__main__":
    debug_dataset()
    
    print("\n" + "="*50)
    print("Starting training with debug...")
    print("="*50)
    
    model, results, losses, events = train_sentence_pnp_oscillator_debug(
        num_sentences=30,
        batch_size=4,
        epochs=30,
        d_model=64,
        nhead=4,
        num_layers=1,
        learning_rate=0.01  # Higher learning rate for faster convergence
    )
    
    print("\nResults Summary:")
    print(f"Model parameters: {results['model_params']:,}")
    print(f"Vocabulary size: {results['vocab_size']}")
    print(f"Dataset size: {results['dataset_size']}")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Compression events: {results['compression_events']}")