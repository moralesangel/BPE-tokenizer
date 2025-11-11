# Byte Pair Encoding (BPE) Tokenizer

A clean, educational implementation of the Byte Pair Encoding algorithm used in modern language models like GPT.

## 📋 Overview

This implementation demonstrates how BPE tokenization works from scratch, including:
- **Training** a vocabulary from text data
- **Encoding** text into token IDs
- **Decoding** token IDs back to text
- **Saving/Loading** trained vocabularies

## 🚀 Quick Start

### Installation

Only requires the HuggingFace datasets library:

```bash
pip install datasets tqdm
```

### Basic Usage

```python
from tokenizer import Tokenizer

# Load pre-trained vocabulary and merges
with open("data/vocab.txt", "r") as f:
    vocab = [line.strip() for line in f]

with open("data/merges.txt", "r") as f:
    merges = [tuple(line.strip().split()) for line in f]

# Initialize tokenizer
tokenizer = Tokenizer(vocab, merges)

# Encode text
text = "Hello world"
token_ids = tokenizer.tokenize(text)
print(f"Tokens: {token_ids}")

# Decode back
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")
```

## 📚 How It Works

### 1. Initial Vocabulary
Start with individual characters as the base vocabulary:

```python
vocab = ['a', 'b', 'c', ..., '</w>']  # </w> marks word boundaries
```

### 2. Iterative Merging
The algorithm repeatedly:
1. **Counts** the most frequent adjacent character pair
2. **Merges** them into a new token
3. **Adds** the new token to the vocabulary

Example progression:
```
"the" → ['t', 'h', 'e', '</w>']
      → ['th', 'e', '</w>']       # Merge 't'+'h'
      → ['the', '</w>']           # Merge 'th'+'e'
      → ['the</w>']               # Merge 'the'+'</w>'
```

### 3. Training Process

```python
def train_bpe(words, vocab_size):
    # Start with character-level tokens
    word_tokens = [list(word + '</w>') for word in words]
    
    # Merge until reaching target vocab size
    while len(vocab) < vocab_size:
        # Find most frequent pair
        pair_counts = count_pairs(word_tokens)
        best_pair = max(pair_counts)
        
        # Merge pair in all words
        word_tokens = merge_pair(word_tokens, best_pair)
        vocab.append(best_pair[0] + best_pair[1])
```

## 🔧 Key Features

### Efficient Tokenization
- Subword units balance vocabulary size and coverage
- Handles out-of-vocabulary words by falling back to characters
- Preserves word boundaries with `</w>` marker

### Memory Efficient
- Stores only vocabulary and merge operations
- No need to save full token-to-ID mappings

### Reproducible
- Deterministic merge order ensures consistent results
- Saved merge history allows exact reconstruction

## 📁 File Structure

```
bpe-tokenizer/
├── bpe_tokenizer.py          # Main tokenizer implementation
├── README.md            # This file
└── data/
    ├── vocab.txt        # Trained vocabulary
    └── merges.txt       # Merge operations history
```

## 💡 Example: Training on Custom Data

```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("your-dataset")
text = " ".join(dataset["train"]["text"])

# Prepare text
words = [word.lower() + '</w>' for word in text.split()]
word_tokens = [list(word) for word in words]

# Train BPE
vocab_size = 4000
vocab, merges = train_bpe(word_tokens, vocab_size)

# Save vocabulary and merges
save_vocab("data/vocab.txt", vocab)
save_merges("data/merges.txt", merges)
```

## 🎯 Configuration

Key parameters to adjust:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `vocab_size` | Target vocabulary size | 1,000 - 50,000 |
| `fraction` | Dataset fraction to use | 0.1 - 1.0 |

### Vocabulary Size Trade-offs

- **Small (1K-5K)**: Faster training, more tokens per word
- **Medium (10K-20K)**: Balanced performance
- **Large (30K-50K)**: Fewer tokens, slower training

## 📊 Performance Notes

**Training Time**:
- 4K vocabulary: ~15 minutes
- 10K vocabulary: ~45 minutes  
- 30K vocabulary: ~3 hours

**Memory Usage**:
- Vocabulary: ~50KB per 1K tokens
- Merge history: ~100KB per 1K merges

## 🔍 Understanding the Output

### Vocabulary File (`vocab.txt`)
```
!
"
a
b
...
th
the
the</w>
```

### Merges File (`merges.txt`)
```
t h
th e
the </w>
...
```
Each line represents a merge operation applied during training.

## 🐛 Common Issues

### Issue: Tokenization produces too many tokens
**Solution**: Increase vocabulary size or train on more data

### Issue: Unknown characters in output
**Solution**: Ensure all characters in test data were in training data

### Issue: Slow tokenization
**Solution**: Pre-compile frequently used merges or use caching

## 📖 Further Reading

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - Original BPE paper
- [HuggingFace Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
- [OpenAI GPT-2 BPE Implementation](https://github.com/openai/gpt-2)

## 📝 License

MIT License - Feel free to use for educational or commercial purposes

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add vocabulary size optimization
- Implement parallel processing
- Add more encoding/decoding options
- Create visualization tools

## ✨ Acknowledgments

Based on the Byte Pair Encoding algorithm introduced by Sennrich et al. (2016) for neural machine translation.
