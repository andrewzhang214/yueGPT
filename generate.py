import torch
import argparse
from yueGPT import yueGPT


def __main__():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read txt file into string
    text = ""
    with open('data/text.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # Get token list and vocab_size
    tokenList = set(text)

    # Create a decoder/encoder
    # This is in lieu of having a proper tokenizer, since we are treating each character
    # as an individual token
    stoi = {t:i for i, t in enumerate(sorted(tokenList))}

    m = yueGPT().to(device=device)
    m.load_state_dict(torch.load('checkpoints/yueGPT28M.pt', map_location=torch.device('cpu')))
    startWord = (torch.ones((1,1), dtype=torch.long) * stoi[' ']).to(device=device)
    m.generate(startWord, max_new_tokens=args.length)


 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text from yueGPT')
    parser.add_argument('-l', '--length', help="Length of text to generate", type=int)
    args = parser.parse_args()
    __main__()