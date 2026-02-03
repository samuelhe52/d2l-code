from utils.rnn import Vocab
from utils.rnn.book_data import TimeMachineData, PrideAndPrejudiceData


if __name__ == "__main__":
    data = PrideAndPrejudiceData(use_chars=False)
    print(f'Number of tokens: {len(data.tokens)}')
    print(f'Vocabulary size: {len(data.vocab)}')
    print(f'Most frequent tokens: {data.vocab.token_freqs[:10]}')
    print(f'Least frequent tokens: {data.vocab.token_freqs[-10:]}')
