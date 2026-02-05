import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Optional, Tuple
from utils.training import RNNTrainer, TrainingConfig
from utils.data import (
    Vocab,
    book_data_loader,
    PrideAndPrejudiceData,
    TimeMachineData,
)
    
class RNNScratch(nn.Module):
    """
    The RNN model implemented from scratch.

    Args:
        vocab_size (int): Size of the vocabulary. \
            Used for input and output dimensions.
        num_hiddens (int): Number of hidden units.
        sigma (float): Standard deviation for weight initialization.
    """
    def __init__(self, vocab_size: int,
                 num_hiddens: int, sigma: float = 0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        # Weights
        self.W_xh = nn.Parameter(
            torch.randn(vocab_size, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * sigma)
        self.W_hq = nn.Parameter(
            torch.randn(num_hiddens, vocab_size) * sigma)
        # Biases
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))
        self.b_q = nn.Parameter(torch.zeros(vocab_size))
        
    def rnn_step(self, X: Tensor, H: Tensor) -> Tensor:
        """
        Perform a single time step of RNN.

        Args:
            X (Tensor): Input tensor at current time step of shape (batch_size, vocab_size).
            H (Tensor): Hidden state tensor from previous time step of shape (batch_size, num_hiddens).
        Returns:
            Tensor: Updated hidden state tensor of shape (batch_size, num_hiddens).
        """
        H = torch.tanh(
            torch.matmul(X, self.W_xh) +
            torch.matmul(H, self.W_hh) + self.b_h
        )
        return H
    
    def rnn_forward(self, inputs: Tensor, state: Optional[Tensor] = None) \
        -> Tuple[Tensor, Tensor]:
        """
        Perform forward pass through the RNN for all time steps.

        Args:
            inputs (Tensor): Input tensor of shape (seq_len, batch_size, vocab_size).
            state (Optional[Tensor]): Initial hidden state tensor of shape (batch_size, num_hiddens).
        Returns:
            Tuple[Tensor, Tensor]:
                - Output tensor of shape (seq_len, batch_size, num_hiddens).
                - Final hidden state tensor of shape (batch_size, num_hiddens).
        """
        if state is None:
            batch_size = inputs.shape[1]
            state = torch.zeros(
                (batch_size, self.num_hiddens),
                device=inputs.device, dtype=inputs.dtype
            )
        H = []
        for X in inputs:
            state = self.rnn_step(X, state)
            H.append(state)
        return torch.stack(H), state

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the RNN model.

        Args:
            inputs (Tensor): Input tensor of shape (batch_size, seq_len).
        Returns:
            Tuple[Tensor, Tensor]:
                - Output tensor of shape (batch_size, vocab_size, seq_len).
                - Final hidden state tensor of shape (batch_size, num_hiddens).
        """
        X = self.one_hot(inputs, self.vocab_size)
        rnn_outputs, H = self.rnn_forward(X, state)
        output = self.output_layer(rnn_outputs)
        return output, H
        
    def output_layer(self, H: Tensor) -> Tensor:
        """
        Compute the output layer.

        Args:
            H (Tensor): Hidden states tensor of shape (seq_len, batch_size, num_hiddens).
        Returns:
            Tensor: Output tensor of shape (batch_size, vocab_size, seq_len). Contains \
                the logits at each time step.
        """
        outputs = [
            torch.matmul(h, self.W_hq) + self.b_q for h in H
        ] # List of (batch_size, vocab_size)
        return torch.stack(outputs, 2) # (batch_size, vocab_size, seq_len)
    
    def one_hot(self, X: Tensor, vocab_size: int) -> Tensor:
        """
        Convert input indices to one-hot encoded vectors.

        Args:
            X (Tensor): Input tensor of shape (batch_size, seq_len).
            vocab_size (int): Size of the vocabulary.
        Returns:
            Tensor: One-hot encoded tensor of shape (seq_len, batch_size, vocab_size).
        """
        return F.one_hot(X.T, num_classes=vocab_size).to(dtype=torch.float32, device=X.device)
    
    def generate(self, prefix: str, num_preds: int,
                 vocab: Vocab, device: torch.device) -> str:
        """
        Generate text using the trained RNN model.

        Args:
            prefix (str): Initial string to start the generation.
            num_preds (int): Number of tokens to generate.
            vocab (Vocab): Vocabulary object for token-index mapping.
            device (torch.device): Device to run the model on.
        Returns:
            str: Generated text.
        """
        state, outputs = None, [vocab[prefix[0]]]
        for t in range(num_preds + len(prefix) - 1):
            X = torch.tensor([[outputs[-1]]], device=device) # The last token
            embs = self.one_hot(X, self.vocab_size) # (1, 1, vocab_size)
            H, state = self.rnn_forward(embs, state) # Save state for next step
            if t < len(prefix) - 1: # Warm-up period
                outputs.append(vocab[prefix[t + 1]]) # Next char from prefix
            else:
                Y = self.output_layer(H) # (1, vocab_size, 1)
                outputs.append(int(Y.argmax(dim=1).squeeze().cpu().numpy()))
        return ''.join([vocab.idx_to_token[i] for i in outputs])
    
if __name__ == "__main__":
    data = TimeMachineData(seq_len=32, use_chars=True)
    train_loader = book_data_loader(
        data, batch_size=1024, train=True
    )
    val_loader = book_data_loader(
        data, batch_size=1024, train=False
    )
    model = RNNScratch(
        vocab_size=len(data.vocab),
        num_hiddens=32,
    )
    
    config = TrainingConfig(
        num_epochs=100,
        lr=1,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device('cpu')
    )
    
    trainer = RNNTrainer(model, train_loader, val_loader, config)
    trainer.train()