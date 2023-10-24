import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from yueGPT import yueGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def __main__():
    m = yueGPT().to(device=device)
    m.load_state_dict(torch.load('checkpoints/yueGPT28M.pt', map_location=torch.device('cpu')))
    startWord = (torch.ones((1,1), dtype=torch.long) * stoi[' ']).to(device=device)
    m.generate(startWord)


if __name__ == "__main__":
    __main__()