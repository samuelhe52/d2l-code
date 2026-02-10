import math
import torch
import time
import matplotlib.pyplot as plt
from utils.data.mt_data import (
    FraEngDataset,
    GerEngDataset,
    mt_dataloader
)


def plot_vocab_histogram(
    vocab, title: str, bins: int = 50, log_scale: bool = True
) -> None:
    freqs = [freq for _, freq in vocab.token_freqs]
    if log_scale:
        values = [math.log10(freq + 1) for freq in freqs]
        xlabel = "log10(1 + token frequency)"
    else:
        values = freqs
        xlabel = "Token frequency"
    plt.hist(values, bins=bins, color="steelblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")


def plot_length_histogram(lengths, title: str, bins: int = 25) -> None:
    plt.hist(lengths, bins=bins, color="seagreen", edgecolor="black")
    plt.title(title)
    plt.xlabel("Sequence length")
    plt.ylabel("Count")
    
    
def visualize_token_freq():
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plot_vocab_histogram(data.src_vocab, "Source vocab frequency histogram")
    plt.subplot(1, 2, 2)
    plot_vocab_histogram(data.tgt_vocab, "Target vocab frequency histogram")
    plt.tight_layout()
    plt.show()
    
def visualize_seq_len():
    tgt_valid_len = (data.tgt_array != data.tgt_vocab["<pad>"]).sum(dim=1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plot_length_histogram(
        data.src_valid_len.tolist(),
        "Source sequence length histogram"
    )
    plt.subplot(1, 2, 2)
    plot_length_histogram(
        tgt_valid_len.tolist(),
        "Target sequence length histogram"
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Time the data loading process
    start_time = time.time()
    data = GerEngDataset(
        seq_len=25,
        token_min_freq=4
    )
    end_time = time.time()
    print(f"Data loading time: {end_time - start_time:.2f} seconds")
    train_loader = mt_dataloader(data, batch_size=3)
    # print(f"Vocab size (source): {len(data.src_vocab)}")
    # print(f"Vocab size (target): {len(data.tgt_vocab)}")

    print(f"Total training samples: {len(data.src_array)}")
        
    # Tokens with lowest frequency
    print("Lowest frequency source tokens:",
          data.src_vocab.token_freqs[-10:])
    print("Lowest frequency target tokens:",
          data.tgt_vocab.token_freqs[-10:])
    
    # Count for seqs longer than 25 tokens
    long_src_seqs = (data.src_valid_len > 25).sum().item()
    long_tgt_seqs = ((data.tgt_array != data.tgt_vocab["<pad>"]).sum(dim=1) > 25).sum().item()
    print(f"Number of source sequences longer than 25 tokens: {long_src_seqs}")
    print(f"Number of target sequences longer than 25 tokens: {long_tgt_seqs}")

    # visualize_token_freq()
    # visualize_seq_len()

    # for ((X, tgt_array, X_valid_len), y_label) in train_loader:
    #     print('X:', X)
    #     print('Y:', tgt_array)
    #     print('X_valid_len:', X_valid_len)
    #     print('Y_label:', y_label)
    #     print('X example (decoded):',
    #             ' '.join([data.src_vocab.idx_to_token[idx.item()] for idx in X[0]]))
    #     print('Y example (decoded):',
    #             ' '.join([data.tgt_vocab.idx_to_token[idx.item()] for idx in tgt_array[0]]))
    #     break