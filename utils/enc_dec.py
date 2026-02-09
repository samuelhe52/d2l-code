import torch
from torch import nn, Tensor
from typing import Tuple
from abc import ABC, abstractmethod

class Encoder(ABC, nn.Module):
    @abstractmethod
    def forward(self, X: Tensor, *args) -> Tuple[Tensor, Tensor]:
        """
        Perform the forward pass of the encoder.

        Args:
            X (Tensor): Input tensor.
            *args: Additional arguments for specific encoder implementations.

        Returns:
            Tuple[Tensor, Tensor]: Encoded output tensor and hidden state.
        """
        pass

class Decoder(ABC, nn.Module):
    @abstractmethod
    def preprocess_state(self, enc_outputs: Tensor, *args) -> Tensor:
        """
        Preprocess the encoder outputs to initialize the decoder state.

        Args:
            enc_outputs (Tensor): Encoder output tensor.
            *args: Additional arguments for specific decoder implementations.

        Returns:
            Tensor: Initial decoder state tensor.
        """
        pass
    
    @abstractmethod
    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Perform the forward pass of the decoder.

        Args:
            X (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Decoded output tensor and hidden state.
        """
        pass
    
class EncoderDecoder(nn.Module):
    """Base class for encoder-decoder models."""
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X: Tensor, dec_X: Tensor, *args) -> Tensor:
        """
        Perform the forward pass of the encoder-decoder model.

        Args:
            enc_X (Tensor): Input tensor for the encoder.
            dec_X (Tensor): Input tensor for the decoder.
            *args: Additional arguments for specific encoder-decoder implementations.

        Returns:
            Tensor: Output tensor from the encoder-decoder model.
        """
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.preprocess_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)[0] # Ignore the hidden state output

    def generate(
        self,
        batch: Tuple,
        device: torch.device,
        max_len: int | None = None,
        save_attention_weights: bool = False,
    ) -> Tuple[Tensor, list[Tensor]]:
        """
        Generate target sequences from a batch using greedy decoding.

        Args:
            batch: A batch in the format returned by mt_data.py:
                ((src_array, tgt_array, src_valid_len), label_array).
            device: Device to run inference on.
            max_len: Optional max decoding length. Defaults to target length.
            save_attention_weights: If True and the decoder exposes
                `attention_weights`, returns them alongside predictions.

        Returns:
            Tuple of (predictions, attention_weights). If
            `save_attention_weights` is False, attention weights will be
            an empty list.
        """
        was_training = self.training
        self.eval()

        (src_array, tgt_array, src_valid_len), _ = batch
        src_array = src_array.to(device)
        src_valid_len = src_valid_len.to(device)
        tgt_array = tgt_array.to(device)

        dec_X = tgt_array[:, :1]
        preds: list[Tensor] = []
        attn_weights: list[Tensor] = []

        decode_len = max_len if max_len is not None else tgt_array.shape[1]

        with torch.no_grad():
            enc_outputs = self.encoder(src_array, src_valid_len)
            dec_state = self.decoder.preprocess_state(enc_outputs, src_valid_len)
            for _ in range(decode_len):
                Y, dec_state = self.decoder(dec_X, dec_state)
                # Y shape: (batch_size, vocab_size, 1)
                pred = Y.argmax(dim=1).squeeze(1)
                preds.append(pred)
                if save_attention_weights and hasattr(self.decoder, "attention_weights"):
                    attn_weights.append(self.decoder.attention_weights)
                dec_X = pred.unsqueeze(1)

        if was_training:
            self.train()

        pred_seq = torch.stack(preds, dim=1)
        if not save_attention_weights:
            attn_weights = []
        return pred_seq, attn_weights
