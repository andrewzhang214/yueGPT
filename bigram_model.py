import os

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


"""
Import dataset - Ancient Chiense Text (文言文) from: https://www.kaggle.com/datasets/raynardj/zh-wenyanwen-wikisource
This dataset is QUITE large, extract only NUM_ROWS from it and concatenate it into a string. If you want your own
uncomment the code below. Otherwise, the data is extracted from the text.txt file.
"""
# df = pd.read_csv("cn_wenyan.csv")

# NUM_ROWS = 500
# print(len(df["text"]))
# textCol = df["text"].head(NUM_ROWS)


# Randomly select from the rows and add the string to the text string to
# create our dataset
# random.seed(1337)
# text = ""
# for _ in range(len(textCol)):
#     r = random.choice(range(len(textCol)))
#     text += textCol[r].replace('</onlyinclude>', '') + '\n'


# HYPERPARAMETERS --------------------------------------
BATCH_SIZE = 32
BLOCK_SIZE = 8 # AKA Context Length
MAX_ITERS = 30000
LEARNING_RATE = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
EVAL_INTERVAL = 200

# ------------------------------------------------------
torch.manual_seed(1337)

# Read txt file into string
text = ""
with open('text.txt', 'r', encoding='utf-8') as file:
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

# Create datasets
data = torch.tensor(encode(text))
n = int(0.9*len(text))
train = data[:n]
eval = data[n:]


# Dataloading (This is in lieu of Dataloader object since
# this dataset is relatively easy to handle)
def get_batch(split):
    # generate batch of data of inputs x and y
    data = train if split == 'train' else eval
    starting_index = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[si:si+BLOCK_SIZE] for si in starting_index])
    y = torch.stack([data[si+1:si+BLOCK_SIZE+1] for si in starting_index])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    # Estimate loss of current model state over random samples
    # from train and eval dataset
    out = {}
    m.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

# Super baby bigram language model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        # inputs are (B, T)
        # targets are (B, T)
        logits = self.embedding_layer(inputs) # B, T, H (Where H = vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, H = logits.shape
            logits = logits.view(B*T, H)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def _generate(self, idx, max_new_tokens):
        # idx is inputs to forward of shape (B, T)
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # logits is of shape (B, T, H)
            # look only at last token in logits
            logits = logits[:, -1, :] # (B, H)
            # Take softmax over H dimension
            probs = F.softmax(logits, -1) # (B)
            # Sample from distribution
            next_idx = torch.multinomial(probs, 1) # (B, 1)
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)
        return idx

    def generate_text(self, max_num_tokens):
        """
        Used for when you want to print out the resulting text, handles decoding as well
        Always starts with '吾'
        """
        text = ""
        startWord = (torch.ones((1,1), dtype=torch.long) * stoi['吾'])
        generated = self._generate(startWord, max_num_tokens)
        generated = decode(generated[0].numpy())  
        for token in generated:
            text += token
        print(text)

# Create model
m = BigramLanguageModel(VOCAB_SIZE).to(device=device)


# Optimization
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

# Training loop: May take a few minutes
for step in range(MAX_ITERS):
    # every EVAL_INTERVAL, evaluate the loss on train and val sets
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss is {losses['train']:.4f}, eval loss is {losses['eval']:.4f}")

    # Randomly sample from training data
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


m.generate_text(2000)

# Save model params to checkpoint file
CWD = os.getcwd()
PATH = CWD + '/bigram40000.pt'
torch.save(m.state_dict(), PATH)


