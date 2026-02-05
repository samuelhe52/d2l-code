import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple, Optional

from utils import load_model
from utils import TrainingConfig
from utils.rnn import Vocab
from utils.rnn.book_data import \
    book_data_loader, TimeMachineData, PrideAndPrejudiceData
from utils.training import RNNTrainer

class RNNLM(nn.Module):
    """
    A simple RNN-based Language Model using PyTorch's built-in RNN module.

    Args:
        vocab_size (int): Size of the vocabulary. \
            Used for input and output dimensions.
        num_hiddens (int): Number of hidden units.
        num_layers (int): Number of RNN layers.
        dropout (float): Dropout probability between RNN layers.
    """
    def __init__(self, vocab_size: int,
                 num_hiddens: int, num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.rnn = nn.RNN(input_size=vocab_size,
                          hidden_size=num_hiddens,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=False)
        self.linear = nn.Linear(num_hiddens, vocab_size)

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None) \
        -> Tuple[Tensor, Tensor]:
        """
        Perform forward pass through the RNN for all time steps.

        Args:
            inputs (Tensor): Input tensor of shape (batch_size, seq_len). \
                One-hot encoding will be applied internally.
            state (Optional[Tensor]): Initial hidden state tensor of shape \
                (num_layers, batch_size, num_hiddens).
        Returns:
            Tuple[Tensor, Tensor]:
                - Output tensor of shape (batch_size, vocab_size, seq_len).
                - Final hidden state tensor of shape (num_layers, batch_size, num_hiddens).
        """
        if state is None:
            batch_size = inputs.shape[0]
            num_layers = self.rnn.num_layers
            state = torch.zeros(
                (num_layers, batch_size, self.num_hiddens),
                device=inputs.device, dtype=torch.float32
            )
        inputs = self.one_hot(inputs)
        rnn_outputs, H = self.rnn(inputs, state)
        return self.output_layer(rnn_outputs), H

    def output_layer(self, rnn_outputs: Tensor) -> Tensor:
        """
        Apply the output linear layer to RNN outputs.

        Args:
            rnn_outputs (Tensor): RNN output tensor of shape \
                (seq_len, batch_size, num_hiddens).
        Returns:
            Tensor: Output tensor of shape (batch_size, vocab_size, seq_len).
        """
        seq_len, batch_size, _ = rnn_outputs.shape
        merged = rnn_outputs.reshape(-1, rnn_outputs.shape[-1])
        outputs = self.linear(merged)
        # Restore (seq_len, batch_size, vocab_size) before putting batch first
        outputs = outputs.reshape(seq_len, batch_size, self.vocab_size)
        return outputs.permute(1, 2, 0)
    
    def one_hot(self, X: Tensor) -> Tensor:
        """
        Convert input indices to one-hot encoded vectors.        

        Args:
            X (Tensor): Input tensor of shape (batch_size, seq_len).
        Returns:
            Tensor: One-hot encoded tensor of shape \
                (seq_len, batch_size, vocab_size).
        """
        return F.one_hot(X.T, num_classes=self.vocab_size) \
            .to(dtype=torch.float32, device=X.device)
            
    def generate(self, prefix: str, num_preds: int,
                 vocab: Vocab, device: torch.device) -> str:
        """
        Generate text using the RNN model

        Args:
            prefix (str): The starting string.
            num_preds (int): Number of tokens to predict.
            vocab (Vocab): Vocabulary object for mapping between tokens and indices.
            device (torch.device): Device on which the computation will be performed.

        Returns:
            str: Generated text string.
        """
        state, outputs = None, [vocab[prefix[0]]]
        for t in range(num_preds + len(prefix) - 1):
            X = torch.tensor([[outputs[-1]]], device=device) # The last token
            Y, state = self.forward(X, state)
            if t < len(prefix) - 1:
                outputs.append(vocab[prefix[t + 1]]) # Append the next token from prefix
            else:
                outputs.append(int(Y.argmax(dim=1).item()))
        return ''.join([vocab.idx_to_token[i] for i in outputs])

if __name__ == "__main__":
    data = TimeMachineData(seq_len=32, use_chars=True)
    train_loader = book_data_loader(
        data, batch_size=1024, train=True
    )
    val_loader = book_data_loader(
        data, batch_size=1024, train=False
    )
    model = RNNLM(
        vocab_size=len(data.vocab),
        num_hiddens=32,
    )
    
    config = TrainingConfig(
        num_epochs=100,
        lr=1,
        loss_fn=nn.CrossEntropyLoss(),
        save_path='./models/rnnlm.pt',
        device=torch.device('cpu')
    )
    
    # trainer = RNNTrainer(model, train_loader, val_loader, config)
    # trainer.train()
    
    # Test generation
    model = load_model('./models/rnnlm.pt', model)
    print(model.generate(
        prefix='time traveller ',
        num_preds=50,
        vocab=data.vocab,
        device=torch.device('cpu')
    ))
