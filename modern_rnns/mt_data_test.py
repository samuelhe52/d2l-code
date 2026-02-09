import torch
from utils.data.mt_data import (
    FraEngDataset,
    GerEngDataset,
    mt_dataloader
)

if __name__ == "__main__":
    data = GerEngDataset(
        seq_len=25,
        token_min_freq=5
    )
    train_loader = mt_dataloader(data, batch_size=3)
    print(f"Vocab size (source): {len(data.src_vocab)}")
    for ((X, tgt_array, X_valid_len), y_label) in train_loader:
        print('X:', X)
        print('Y:', tgt_array)
        print('X_valid_len:', X_valid_len)
        print('Y_label:', y_label)
        print('X example (decoded):',
                ' '.join([data.src_vocab.idx_to_token[idx.item()] for idx in X[0]]))
        print('Y example (decoded):',
                ' '.join([data.tgt_vocab.idx_to_token[idx.item()] for idx in tgt_array[0]]))
        break