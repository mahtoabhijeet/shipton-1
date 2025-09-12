import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
from model import MemoryEfficientTransformer, SentenceTokenizer, create_sentence_dataset, train_model, evaluate_model, get_model_info

def generate_sample_texts(num_samples=1000):
    """Generate sample texts for training."""
    subjects = [
        "the cat", "the dog", "the bird", "the fish", "the rabbit", 
        "john", "mary", "the child", "the teacher", "the student",
        "a man", "a woman", "the boy", "the girl", "the person",
        "the car", "the house", "the tree", "the book", "the computer"
    ]
    
    verbs = [
        "runs", "jumps", "sleeps", "eats", "plays", "swims", "flies", 
        "reads", "writes", "sings", "dances", "walks", "talks", "laughs",
        "thinks", "works", "studies", "cooks", "drives", "builds"
    ]
    
    objects = [
        "in the park", "on the mat", "under the tree", "in the house", 
        "at school", "with a ball", "in the water", "on the roof", 
        "with a book", "very well", "quickly", "slowly", "loudly",
        "in the garden", "at home", "with friends", "every day",
        "in the morning", "at night", "with joy"
    ]
    
    adjectives = [
        "big", "small", "red", "blue", "happy", "sad", "fast", 
        "slow", "loud", "quiet", "young", "old", "smart", "funny",
        "beautiful", "ugly", "strong", "weak", "brave", "shy"
    ]
    
    adverbs = [
        "quickly", "slowly", "carefully", "loudly", "quietly",
        "happily", "sadly", "angrily", "gently", "roughly"
    ]
    
    texts = []
    
    # Pattern 1: Simple subject + verb
    for _ in range(num_samples // 8):
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        texts.append(f"{subject} {verb}")
    
    # Pattern 2: Subject + verb + object
    for _ in range(num_samples // 8):
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        texts.append(f"{subject} {verb} {obj}")
    
    # Pattern 3: Subject + adjective + verb
    for _ in range(num_samples // 8):
        subject = random.choice(subjects)
        adj = random.choice(adjectives)
        verb = random.choice(verbs)
        texts.append(f"{subject} is {adj} and {verb}")
    
    # Pattern 4: Adjective + subject + verb
    for _ in range(num_samples // 8):
        adj = random.choice(adjectives)
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        texts.append(f"{adj} {subject} {verb}")
    
    # Pattern 5: Subject + verb + adverb
    for _ in range(num_samples // 8):
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        adv = random.choice(adverbs)
        texts.append(f"{subject} {verb} {adv}")
    
    # Pattern 6: Complex sentence with conjunction
    for _ in range(num_samples // 8):
        subject1 = random.choice(subjects)
        verb1 = random.choice(verbs)
        subject2 = random.choice(subjects)
        verb2 = random.choice(verbs)
        texts.append(f"{subject1} {verb1} and {subject2} {verb2}")
    
    # Pattern 7: Subject + verb + object + adverb
    for _ in range(num_samples // 8):
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        adv = random.choice(adverbs)
        texts.append(f"{subject} {verb} {obj} {adv}")
    
    # Pattern 8: Adjective + subject + verb + object
    for _ in range(num_samples // 8):
        adj = random.choice(adjectives)
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        texts.append(f"{adj} {subject} {verb} {obj}")
    
    return texts

def load_real_texts():
    """Load a small sample of real texts for demonstration."""
    # This is a small sample - in practice you would load from files
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "a journey of a thousand miles begins with a single step",
        "to be or not to be that is the question",
        "in the middle of difficulty lies opportunity",
        "the only way to do great work is to love what you do",
        "life is what happens when you are busy making other plans",
        "the future belongs to those who believe in the beauty of their dreams",
        "it does not matter how slowly you go as long as you do not stop",
        "everything you have ever wanted is on the other side of fear",
        "success is not final failure is not fatal it is the courage to continue that counts",
        "the greatest glory in living lies not in never falling but in rising every time we fall",
        "the way to get started is to quit talking and begin doing",
        "your time is limited so do not waste it living someone else life",
        "if life were predictable it would cease to be life and be without flavor",
        "if you look at what you have in life you will always have more",
        "if you set your goals ridiculously high and it is a failure you will fail above everyone else success",
        "life is what happens to you while you are busy making other plans",
        "the future belongs to those who prepare for it today",
        "the purpose of our lives is to be happy",
        "go confidently in the direction of your dreams live the life you have imagined"
    ]
    return texts

def main():
    """Main training function."""
    print("P/NP Oscillating Transformer - Sentence Processing")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check for GPU availability
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate or load texts
    print("Loading/Generating texts...")
    # For demo, we'll use a mix of generated and real texts
    generated_texts = generate_sample_texts(500)
    real_texts = load_real_texts()
    all_texts = generated_texts + real_texts
    print(f"Total texts: {len(all_texts)}")
    
    # Create tokenizer and build vocabulary
    print("Building vocabulary...")
    tokenizer = SentenceTokenizer(max_vocab_size=5000)
    tokenizer.build_vocab(all_texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset
    print("Creating dataset...")
    X, y = create_sentence_dataset(all_texts, tokenizer, max_length=50)
    print(f"Dataset shape: {X.shape}, {y.shape}")
    
    # Create data loader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    print("Creating model...")
    model = MemoryEfficientTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        nhead=8,
        num_layers=4,
        max_seq_length=50
    )
    
    # Print model info
    model_info = get_model_info(model)
    print(f"Model parameters: {model_info['total_params']:,}")
    print(f"Estimated memory usage: {model_info['memory_estimate_mb']:.1f} MB")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Train model
    losses, events = train_model(model, train_loader, optimizer, criterion, device, epochs=20)
    
    # Print results
    print(f"\nTraining completed!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Total compression events: {len(events)}")
    
    # Show sample compression events
    if events:
        print("\nSample compression events:")
        for event in events[:5]:
            print(f"  Step {event['step']:3d} | Rank {event['rank_before']:4.1f} → {event['rank_after']:4.1f} | "
                  f"Phase {event['phase']:.2f}")
    
    # Evaluate model
    test_sentences = [
        "the cat",
        "the dog runs",
        "the bird is happy and",
        "a smart student",
        "the quick brown"
    ]
    
    evaluate_model(model, test_sentences, tokenizer, device)
    
    # Save model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'model_info': model_info
    }, 'pn_transformer_model.pth')
    print("Model saved as 'pn_transformer_model.pth'")
    
    return model, tokenizer, losses, events

if __name__ == "__main__":
    model, tokenizer, losses, events = main()