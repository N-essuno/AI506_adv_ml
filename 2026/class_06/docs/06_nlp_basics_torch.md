# Natural Language Processing Basics in PyTorch

---

## Background

In this exercise you will build the core components of an NLP pipeline from scratch using PyTorch. Starting from raw text, you will implement tokenization, construct a vocabulary, map tokens to dense embeddings, and feed the resulting representations into two different model architectures: an MLP (with pooling) and an RNN.

The dataset used throughout is a small collection of movie-review sentences labelled as positive (1) or negative (0).

```python
sentences = [
    "the film was wonderful and touching",
    "terrible movie very boring and slow",
    "great performances and a beautiful story",
    "awful acting the plot made no sense",
    "loved every moment of this masterpiece",
    "waste of time poorly written script",
]
labels = [1, 0, 1, 0, 1, 0]
```

---

## Part 1 — Tokenization

### 1a. Whitespace tokenizer

Implement a function `whitespace_tokenize(text: str) -> list[str]` that lowercases the input and splits on whitespace.

```python
def whitespace_tokenize(text: str) -> list[str]:
    # TODO
    pass
```

**Expected output** for `"The film was GREAT"`:
```
['the', 'film', 'was', 'great']
```

### 1b. Character-level tokenizer

Implement `char_tokenize(text: str) -> list[str]` that lowercases the input and returns one token per character, **excluding spaces**.

```python
def char_tokenize(text: str) -> list[str]:
    # TODO
    pass
```

**Expected output** for `"Hi NLP"`:
```
['h', 'i', 'n', 'l', 'p']
```

### 1c. Discussion

Compare whitespace tokenization and character-level tokenization on the following dimensions. Write 2–3 sentences for each.

1. **Vocabulary size**: how does it scale with corpus size?
2. **Out-of-vocabulary (OOV) handling**: what happens with unseen words?
3. **Sequence length**: how does the average sequence length differ between the two?

---

## Part 2 — Vocabulary

### 2a. Build vocabulary

Implement the class below. The vocabulary must include two special tokens: `<PAD>` (index 0) and `<UNK>` (index 1). All other tokens are assigned consecutive indices in the order they are first encountered.

```python
class Vocabulary:
    def __init__(self):
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}

    def build(self, tokenized_corpus: list[list[str]]) -> None:
        """Populate token2idx and idx2token from a list of tokenized sentences."""
        # TODO
        pass

    def encode(self, tokens: list[str]) -> list[int]:
        """Map each token to its index; unknown tokens map to <UNK>."""
        # TODO
        pass

    def decode(self, indices: list[int]) -> list[str]:
        """Map each index back to its token string."""
        # TODO
        pass

    def __len__(self) -> int:
        return len(self.token2idx)
```

### 2b. Padding

Implement `pad_sequences(sequences: list[list[int]], pad_idx: int = 0) -> torch.Tensor` that pads all sequences to the length of the longest one and returns a `Tensor` of shape `(batch_size, max_len)`.

```python
import torch

def pad_sequences(sequences: list[list[int]], pad_idx: int = 0) -> torch.Tensor:
    # TODO
    pass
```

**Expected output** for `[[1, 2, 3], [4, 5]]`:
```
tensor([[1, 2, 3],
        [4, 5, 0]])
```

---

## Part 3 — Embeddings

### 3a. Embedding layer

Using `nn.Embedding`, implement `build_embedding_layer(vocab_size, embed_dim, pad_idx=0)` that returns an embedding layer where the padding vector is always the zero vector and is not updated during training.

```python
import torch.nn as nn

def build_embedding_layer(vocab_size: int,
                           embed_dim: int,
                           pad_idx: int = 0) -> nn.Embedding:
    # TODO
    pass
```

Verify that `emb(torch.tensor([0]))` returns all zeros after the layer is created.

### 3b. Embedding lookup

Given the padded batch tensor from Part 2b, write the code to produce an embedding tensor of shape `(batch_size, max_len, embed_dim)`. Then print the shape and confirm it matches expectations for `batch_size=6`, `max_len=6`, `embed_dim=16`.

---

## Part 4 — MLP with Mean Pooling

An MLP cannot consume sequences of variable length directly. A common solution is to **pool** the token embeddings on the sequence dimension before passing the result to the MLP.

### 4a. Masked mean pooling

Implement `masked_mean_pool(embeddings, attention_mask)` where:

- `embeddings` has shape `(B, T, D)`
- `attention_mask` is a boolean tensor of shape `(B, T)` that is `True` for real tokens and `False` for padding

The function must return a tensor of shape `(B, D)` that is the mean of the **non-padding** embeddings for each sentence.

```python
def masked_mean_pool(embeddings: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    # TODO
    pass
```

> **Hint**: expand the mask to `(B, T, D)` before multiplying, then divide by the count of non-padding tokens per row.

### 4b. MLP classifier

Implement `TextMLP` below. The forward pass must:

1. Look up embeddings.
2. Compute the attention mask (positions where `input_ids != pad_idx`).
3. Apply masked mean pooling.
4. Pass the pooled vector through two linear layers with ReLU activation between them and dropout before the final layer.

```python
class TextMLP(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 pad_idx: int = 0,
                 dropout: float = 0.3):
        super().__init__()
        # TODO: define layers

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T)  →  output: (B, num_classes)
        # TODO
        pass
```

---

## Part 5 — RNN Classifier

### 5a. RNN-based model

Implement `TextRNN` using a single-layer RNN. The forward pass must:

1. Look up embeddings.
2. Pack the padded sequences using `nn.utils.rnn.pack_padded_sequence` (use the actual sequence lengths, not the padded length).
3. Run the packed sequence through the RNN.
4. Unpack the output using `pad_packed_sequence`.
5. Use the **final hidden state** of the RNN as the sentence representation.
6. Pass it through a single linear layer to produce the class logits.

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TextRNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 pad_idx: int = 0):
        super().__init__()
        # TODO: define layers

    def forward(self,
                input_ids: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T), lengths: (B,)  →  output: (B, num_classes)
        # TODO
        pass
```

Finally create a simple training loop for both `TextMLP` and `TextRNN`.