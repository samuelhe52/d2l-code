from utils.data.book_data import (
    book_data_loader,
    PrideAndPrejudiceData,
    TimeMachineData,
    AustenCompilationData
)


if __name__ == "__main__":
    data = AustenCompilationData(seq_len=10, use_chars=True)
    print(f'Number of tokens: {len(data.tokens)}')
    print(f'Vocabulary size: {len(data.vocab)}')
    print(f'Most frequent tokens: {data.vocab.token_freqs[:10]}')
    print(f'Least frequent tokens: {data.vocab.token_freqs[-10:]}')

    data = TimeMachineData(seq_len=10, use_chars=True)
    train_loader = book_data_loader(
        data, batch_size=2, train=True
    )
    
    for X, Y in train_loader:
        print('X:', X, '\nY:', Y)
        print('X shape:', X.shape)
        print('Y shape:', Y.shape)
        break
