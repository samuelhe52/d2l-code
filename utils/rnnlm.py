from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from utils.data import Vocab


class RNNLM(nn.Module):
    """
    A simple RNN-based Language Model using PyTorch's built-in RNN module.

    Args:
        vocab_size (int): Size of the vocabulary. \
            Used for input and output dimensions.
        num_hiddens (int): Number of hidden units.
        num_layers (int): Number of RNN layers.
        dropout (float): Dropout probability between RNN layers.
        rnn (Optional[nn.Module]): Predefined RNN module. \
            If None, a default nn.RNN will be created.
    """

    def __init__(
        self,
        vocab_size: int,
        num_hiddens: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.rnn = rnn if rnn is not None else nn.RNN(
            input_size=vocab_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=False,
        )
        self.linear = nn.Linear(num_hiddens, vocab_size)

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
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
            if isinstance(self.rnn, nn.LSTM):
                state = (
                    torch.zeros(
                        (num_layers, batch_size, self.num_hiddens),
                        device=inputs.device,
                        dtype=torch.float32,
                    ),
                    torch.zeros(
                        (num_layers, batch_size, self.num_hiddens),
                        device=inputs.device,
                        dtype=torch.float32,
                    ),
                )
            else:
                state = torch.zeros(
                    (num_layers, batch_size, self.num_hiddens),
                    device=inputs.device,
                    dtype=torch.float32,
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
        return F.one_hot(X.T, num_classes=self.vocab_size).to(
            dtype=torch.float32,
            device=X.device,
        )

    def generate(
        self,
        prefix: str,
        num_preds: int,
        vocab: Vocab,
        device: torch.device,
        temperature: float = 1.0,
        top_k: Optional[int] = 20,
    ) -> str:
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
        was_training = self.training
        self.eval()
        state, outputs = None, [vocab[prefix[0]]]
        temp = max(1e-6, float(temperature))
        with torch.no_grad():
            for t in range(num_preds + len(prefix) - 1):
                # The last token
                X = torch.tensor([[outputs[-1]]], device=device)
                Y, state = self.forward(X, state)
                if t < len(prefix) - 1:
                    # Append the next token from prefix
                    outputs.append(vocab[prefix[t + 1]])
                    continue

                logits = Y[:, :, -1] / temp
                if top_k is not None and top_k > 0:
                    k = min(int(top_k), logits.shape[1])
                    topk_vals, topk_idx = torch.topk(logits, k=k, dim=1)
                    probs = F.softmax(topk_vals, dim=1)
                    next_rel = torch.multinomial(probs, num_samples=1)
                    next_idx = topk_idx.gather(1, next_rel)
                else:
                    probs = F.softmax(logits, dim=1)
                    next_idx = torch.multinomial(probs, num_samples=1)
                outputs.append(int(next_idx.item()))

        if was_training:
            self.train()
        return "".join([vocab.idx_to_token[i] for i in outputs])
