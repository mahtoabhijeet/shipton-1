import random
import re

class SentenceTokenizer:
    def __init__(self, max_vocab_size=500):
        self.max_vocab_size = max_vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, sentences):
        # Count word frequencies
        word_counts = {}
        for sentence in sentences:
            # Simple tokenization - split by whitespace and clean punctuation
            words = re.findall(r'\b\w+\b', sentence.lower())
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and limit vocabulary size
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        limited_words = sorted_words[:self.max_vocab_size-4]  # Reserve space for special tokens
        
        # Add special tokens
        self.word_to_idx = {'[PAD]': 0, '[START]': 1, '[END]': 2, '[UNK]': 3}
        self.idx_to_word = {0: '[PAD]', 1: '[START]', 2: '[END]', 3: '[UNK]'}
        
        # Add regular words
        for i, (word, _) in enumerate(limited_words):
            self.word_to_idx[word] = i + 4
            self.idx_to_word[i + 4] = word
            
        self.vocab_size = len(self.word_to_idx)
        
    def encode(self, sentence, max_length=20):
        # Simple tokenization - split by whitespace and clean punctuation
        words = re.findall(r'\b\w+\b', sentence.lower())
        
        # Start with [START] token
        indices = [self.word_to_idx.get('[START]', 1)]
        
        # Add words (or [UNK] for out-of-vocabulary)
        for word in words:
            idx = self.word_to_idx.get(word, self.word_to_idx.get('[UNK]', 3))
            indices.append(idx)
            
        # Add [END] token
        indices.append(self.word_to_idx.get('[END]', 2))
        
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

def generate_sample_sentences(num_sentences=100):
    """
    Generate a sample dataset of simple sentences for training.
    These sentences follow simple patterns to make learning tractable.
    """
    subjects = ["the cat", "the dog", "the bird", "the fish", "the rabbit", 
                "john", "mary", "the child", "the teacher", "the student"]
    
    verbs = ["runs", "jumps", "sleeps", "eats", "plays", "swims", "flies", 
             "reads", "writes", "sings"]
    
    objects = ["in the park", "on the mat", "under the tree", "in the house", 
               "at school", "with a ball", "in the water", "on the roof", 
               "with a book", "very well"]
    
    adjectives = ["big", "small", "red", "blue", "happy", "sad", "fast", 
                  "slow", "loud", "quiet"]
    
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

def create_sentence_dataset(num_sentences=100, max_length=20):
    """
    Create a complete sentence dataset with input sequences and targets.
    """
    # Generate sample sentences
    sentences = generate_sample_sentences(num_sentences)
    
    # Create tokenizer and build vocabulary
    tokenizer = SentenceTokenizer(max_vocab_size=200)
    tokenizer.build_vocab(sentences)
    
    # Create input and target sequences
    input_sequences = []
    target_sequences = []
    
    for sentence in sentences:
        # Encode sentence
        full_sequence = tokenizer.encode(sentence, max_length)
        
        # Input is the sequence without the last token
        input_seq = full_sequence[:-1]
        
        # Target is the sequence without the first token
        target_seq = full_sequence[1:]
        
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    
    return input_sequences, target_sequences, tokenizer, sentences

# Example usage
if __name__ == "__main__":
    # Generate a small dataset for testing
    inputs, targets, tokenizer, sentences = create_sentence_dataset(20, 15)
    
    print("Sample sentences:")
    for i in range(3):
        print(f"{i+1}. {sentences[i]}")
    
    print("\nVocabulary size:", tokenizer.vocab_size)
    print("\nSample encodings:")
    for i in range(3):
        input_seq = inputs[i]
        target_seq = targets[i]
        decoded_input = tokenizer.decode(input_seq)
        decoded_target = tokenizer.decode(target_seq)
        print(f"Input:  {input_seq}")
        print(f"Target: {target_seq}")
        print(f"Decoded input:  '{decoded_input}'")
        print(f"Decoded target: '{decoded_target}'")
        print()