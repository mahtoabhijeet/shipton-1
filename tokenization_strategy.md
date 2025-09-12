# Vocabulary and Tokenization Strategy for Sentence-Based P/NP Oscillator

## Tokenization Approach

For our minimal P/NP oscillator implementation, we'll use a simple word-level tokenization approach:

### 1. Basic Tokenization
- Split sentences by whitespace
- Convert to lowercase for consistency
- Handle punctuation as separate tokens
- No special handling for numbers or special characters

### 2. Vocabulary Design

#### Special Tokens
```
[PAD] - Padding token for uniform sequence length
[START] - Start of sentence marker
[END] - End of sentence marker
[UNK] - Unknown token for out-of-vocabulary words
```

#### Vocabulary Building
- Collect all unique words from training sentences
- Limit vocabulary size to control model complexity
- Assign indices to words (0-indexed)
- Handle out-of-vocabulary words with [UNK] token

### 3. Implementation Strategy

#### Simple Vocabulary Class
```python
class SentenceTokenizer:
    def __init__(self, max_vocab_size=1000):
        self.max_vocab_size = max_vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, sentences):
        # Count word frequencies
        word_counts = {}
        for sentence in sentences:
            for word in sentence.lower().split():
                # Simple cleaning - remove basic punctuation
                cleaned_word = ''.join(c for c in word if c.isalnum())
                if cleaned_word:
                    word_counts[cleaned_word] = word_counts.get(cleaned_word, 0) + 1
        
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
        
    def encode(self, sentence):
        words = sentence.lower().split()
        indices = [self.word_to_idx.get('[START]')]
        
        for word in words:
            # Clean word
            cleaned_word = ''.join(c for c in word if c.isalnum())
            if cleaned_word:
                # Use [UNK] for out-of-vocabulary words
                idx = self.word_to_idx.get(cleaned_word, self.word_to_idx['[UNK]'])
                indices.append(idx)
                
        indices.append(self.word_to_idx.get('[END]'))
        return indices
        
    def decode(self, indices):
        words = []
        for idx in indices:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word not in ['[PAD]', '[START]', '[END]']:
                    words.append(word)
        return ' '.join(words)
```

## Sequence Processing

### Fixed-Length Sequences
To work with our transformer model, we need fixed-length sequences:

#### Padding Strategy
- Pad shorter sequences with [PAD] token
- Truncate longer sequences to maximum length
- Default maximum length: 20 tokens (adjustable)

#### Sequence Structure
```
[START] word1 word2 word3 ... [END] [PAD] [PAD] ...
```

### Target Generation
For next-word prediction:
- Input: [START] word1 word2 word3 ...
- Target: word1 word2 word3 ... [END]

## Example Implementation

### Sample Sentences
```
"The cat sat on the mat"
"I like to eat apples"
"The dog ran in the park"
"She plays piano very well"
```

### Tokenized Sequences (max_len=10)
```
[1, 5, 12, 23, 15, 5, 18, 2, 0, 0]  # "The cat sat on the mat"
[1, 9, 21, 7, 31, 4, 0, 0, 0, 0]    # "I like to eat apples"
```

## Vocabulary Size Considerations

### For Minimal Implementation
- Small vocabulary (100-500 words) for fast training
- Controlled dataset with limited topics
- Focus on demonstrating sentence processing rather than language modeling

### For Extended Implementation
- Larger vocabulary (1000-5000 words) for more realistic text
- Diverse sentence structures
- More complex language patterns

## Memory and Performance Implications

### Embedding Layer
- Size: vocab_size × embedding_dim
- Larger vocabularies require more memory
- For 1000 words with 64-dim embeddings: ~256KB

### Attention Computation
- Complexity: O(sequence_length²)
- Longer sequences increase computation quadratically
- Need to balance sequence length with computational constraints

## Implementation Plan

1. Create tokenizer class with build_vocab, encode, decode methods
2. Implement padding and truncation functionality
3. Design sample sentence dataset
4. Modify model for sentence processing
5. Test with simple examples