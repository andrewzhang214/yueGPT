
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import pandas as pd
import random


"""
Import dataset - Ancient Chiense Text (文言文) from: https://www.kaggle.com/datasets/raynardj/zh-wenyanwen-wikisource
This dataset is QUITE large, extract only NUM_ROWS from it and concatenate it into a string. If you want your own
uncomment the code below. Otherwise, the data is extracted from the text.txt file.
"""

# HYPERPARAMETERS --------------------------------------
BATCH_SIZE = 32
BLOCK_SIZE = 64 # AKA Context Length
MAX_ITERS = 5000
LEARNING_RATE = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
EVAL_INTERVAL = 200
N_EMBED = 512
N_HEAD = 8
N_LAYER = 6
dropout = 0.2
# ------------------------------------------------------

# Read txt file into string
text = ""
with open('data/text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Get token list and vocab_size
tokenList = set(text)
VOCAB_SIZE = len(tokenList)

# Create a decoder/encoder
# This is in lieu of having a proper tokenizer, since we are treating each character
# as an individual token
stoi = {t:i for i, t in enumerate(sorted(tokenList))}
itos = {i:t for i, t in enumerate(sorted(tokenList))}
encode = lambda s: [stoi[t] for t in s]
decode = lambda t: [itos[i] for i in t]


class Head(nn.Module):
    """Single head self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is shape B, T, H
        B, T, H = x.shape
        k = self.key(x) # B, T, HEAD_SIZE
        q = self.key(x) # B, T, HEAD_SIZE
        v = self.key(x) # B, T, HEAD_SIZE

        # Compute attention scores
        weights = q @ k.transpose(-2, -1) * H**-0.5 # (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) # Avoid attention with future tokens # WHY [:T, :T]??
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        out = weights @ v # (B, T, HEAD_SIZE)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention in Parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    """Simple feed forward network after self-attention"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """Transformer decoder block"""
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Res connection
        x = x + self.ffwd(self.ln2(x)) # Res connection
        return x


# yueGPT Model
class yueGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.positional_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE)

    def forward(self, inputs, targets=None):
        # inputs are (B, T)
        # targets are (B, T)
        B, T = inputs.shape
        token_embeddings = self.embedding_layer(inputs) # B, T, H (Where H = embedding size)
        pos_embeddings = self.positional_embedding(torch.arange(T, device=device)) # T, H
        x = token_embeddings + pos_embeddings # B, T, H
        x = self.blocks(x) # B, T, H
        logits = self.lm_head(x) # B, T, V (Where V = VOCAB_SIZE)
        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=None):
        # idx is inputs to forward of shape (B, T)
        if max_new_tokens is not None:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -BLOCK_SIZE:] # (Gets last T entries: (B, T))
                logits, _ = self(idx_cond) # logits is of shape (B, T, H)
                # look only at last token in logits
                logits = logits[:, -1, :] # (B, H)
                # Take softmax over H dimension
                probs = F.softmax(logits, -1) # (B)
                # Sample from distribution
                next_idx = torch.multinomial(probs, 1) # (B, 1)
                print(decode(next_idx[0].cpu().numpy())[0], end='')
                idx = torch.cat((idx, next_idx), dim=1)
        else:
            while True:
                idx_cond = idx[:, -BLOCK_SIZE:] # (Gets last T entries: (B, T))
                logits, _ = self(idx_cond) # logits is of shape (B, T, H)
                # look only at last token in logits
                logits = logits[:, -1, :] # (B, H)
                # Take softmax over H dimension
                probs = F.softmax(logits, -1) # (B)
                # Sample from distribution
                next_idx = torch.multinomial(probs, 1) # (B, 1)
                print(decode(next_idx[0].cpu().numpy())[0], end='')
                idx = torch.cat((idx, next_idx), dim=1)



