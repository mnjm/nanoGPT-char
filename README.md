## nanoGPT-char

A minimal implementation of a character-level GPT model trained on the Tiny Shakespeare dataset, following Andrej Karpathy's [Let's build a GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) video.

Main file is `nano-gpt-char.py`, a small 214 lines of heavily commented code, implements a small GPT model (nanoGPT) and trains it on `input.txt` a tiny toy shakespeare dataset (source: [nanoGPT](https://github.com/karpathy/nanoGPT)) considering individual characters as tokens instead of words/subwords.

Training log can be found in `train.ipynb`, model `gpt-char-model.pth` and sample output `output.txt`

**Model Architecture**

- Embedding dimension: 384
- Number of layers: 6
- Number of attention heads: 6
- Context length: 256 characters
- Batch size: 64
- **Total parameters**: ~10.8M