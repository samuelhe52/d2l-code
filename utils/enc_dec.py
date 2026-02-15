import torch
from torch import nn, Tensor
from typing import Any, Tuple, Optional
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
    def preprocess_state(self, enc_outputs: Tensor, *args) -> Any:
        """
        Preprocess the encoder outputs to initialize the decoder state.

        Args:
            enc_outputs (Tensor): Encoder output tensor.
            *args: Additional arguments for specific decoder implementations.

        Returns:
            Any: Initial decoder state.
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

    def _repeat_state(self, state: Any, repeat: int, batch_size: int) -> Any:
        if torch.is_tensor(state):
            if state.ndim == 0:
                return state
            if state.shape[0] == batch_size:
                return state.repeat_interleave(repeat, dim=0)
            if state.ndim >= 2 and state.shape[1] == batch_size:
                return state.repeat_interleave(repeat, dim=1)
            return state
        if isinstance(state, tuple):
            return tuple(self._repeat_state(s, repeat, batch_size) for s in state)
        if isinstance(state, list):
            return [self._repeat_state(s, repeat, batch_size) for s in state]
        return state

    def _select_state(self, state: Any, indices: Tensor, total_beams: int) -> Any:
        if torch.is_tensor(state):
            if state.ndim >= 2 and state.shape[1] == total_beams:
                return state.index_select(1, indices)
            if state.shape[0] == total_beams:
                return state.index_select(0, indices)
            return state
        if isinstance(state, tuple):
            return tuple(self._select_state(s, indices, total_beams) for s in state)
        if isinstance(state, list):
            return [self._select_state(s, indices, total_beams) for s in state]
        return state

    def _infer_eos(self, label_array: Tensor, pad_idx: Optional[int]) -> Optional[int]:
        if pad_idx is None:
            return None
        non_pad = label_array != pad_idx
        if not torch.any(non_pad):
            return None
        lengths = non_pad.sum(dim=1).clamp_min(1)
        last_idx = lengths - 1
        row_idx = torch.arange(label_array.shape[0], device=label_array.device)
        eos_candidates = label_array[row_idx, last_idx]
        return torch.mode(eos_candidates).values.item()

    def _resolve_eos_index(self, batch: Tuple, device: torch.device) -> Optional[int]:
        eos_token_index = getattr(self, "eos_token_index", None)
        if eos_token_index is not None or len(batch) <= 1:
            return eos_token_index
        label_array = batch[1]
        if torch.is_tensor(label_array):
            return self._infer_eos(
                label_array.to(device),
                getattr(self, "pad_token_index", None),
            )
        return None

    def _greedy_decode(
        self,
        dec_X: Tensor,
        dec_state: Any,
        decode_len: int,
    ) -> Tensor:
        preds: list[Tensor] = []

        for _ in range(decode_len):
            Y, dec_state = self.decoder(dec_X, dec_state)
            pred = Y.argmax(dim=1).squeeze(1)
            preds.append(pred)
            dec_X = pred.unsqueeze(1)

        if preds:
            pred_seq = torch.stack(preds, dim=1)
        else:
            pred_seq = dec_X.new_empty((dec_X.shape[0], 0))
        return pred_seq

    def _beam_decode(
        self,
        dec_X: Tensor,
        dec_state: Any,
        decode_len: int,
        beam_size: int,
        eos_token_index: Optional[int],
        device: torch.device,
        tgt_array: Tensor,
    ) -> Tensor:
        batch_size = dec_X.shape[0]
        vocab_size = self.decoder.vocab_size
        beam_size = min(beam_size, vocab_size)

        dec_state = self._repeat_state(dec_state, beam_size, batch_size)
        dec_X = dec_X.repeat_interleave(beam_size, dim=0)

        scores = torch.full((batch_size, beam_size), -1e9, device=device)
        scores[:, 0] = 0.0
        finished = torch.zeros(
            (batch_size, beam_size), dtype=torch.bool, device=device
        )
        seqs: Optional[Tensor] = None

        for _ in range(decode_len):
            Y, next_state = self.decoder(dec_X, dec_state)
            log_probs = torch.log_softmax(Y.squeeze(2), dim=1)
            log_probs = log_probs.view(batch_size, beam_size, vocab_size)

            if eos_token_index is not None and torch.any(finished):
                log_probs = log_probs.masked_fill(
                    finished.unsqueeze(-1),
                    -1e9,
                )
                eos_scores = log_probs[:, :, eos_token_index]
                log_probs[:, :, eos_token_index] = torch.where(
                    finished,
                    torch.zeros_like(eos_scores),
                    eos_scores,
                )

            total_scores = scores.unsqueeze(-1) + log_probs
            flat_scores = total_scores.view(batch_size, -1)
            topk_scores, topk_indices = flat_scores.topk(beam_size, dim=1)
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            if seqs is None:
                seqs = token_indices.unsqueeze(-1)
            else:
                gather_idx = beam_indices.unsqueeze(-1).expand(
                    -1, -1, seqs.shape[2]
                )
                seqs = torch.gather(seqs, 1, gather_idx)
                seqs = torch.cat([seqs, token_indices.unsqueeze(-1)], dim=2)

            finished = torch.gather(finished, 1, beam_indices)
            if eos_token_index is not None:
                finished = finished | (token_indices == eos_token_index)

            flat_indices = (
                torch.arange(batch_size, device=device) * beam_size
            ).unsqueeze(1) + beam_indices
            flat_indices = flat_indices.reshape(-1)
            dec_state = self._select_state(
                next_state, flat_indices, batch_size * beam_size
            )
            dec_X = token_indices.reshape(-1, 1)
            scores = topk_scores

            if eos_token_index is not None and torch.all(finished):
                break

        best_idx = scores.argmax(dim=1)
        batch_idx = torch.arange(batch_size, device=device)
        if seqs is None:
            pred_seq = torch.empty(
                (batch_size, 0), dtype=tgt_array.dtype, device=device
            )
        else:
            pred_seq = seqs[batch_idx, best_idx]

        if pred_seq.shape[1] < decode_len:
            pad_token = (
                eos_token_index
                if eos_token_index is not None
                else getattr(self, "pad_token_index", 0)
            )
            pad = torch.full(
                (batch_size, decode_len - pred_seq.shape[1]),
                pad_token,
                dtype=pred_seq.dtype,
                device=device,
            )
            pred_seq = torch.cat([pred_seq, pad], dim=1)

        return pred_seq
    
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
        max_len: Optional[int] = None,
        decode_strategy: str = "beam",
        beam_size: int = 4,
    ) -> Tuple[Tensor, list[Tensor]]:
        """
        Generate target sequences from a batch, using beam search decoding by default.

        Args:
            batch: A batch in the format returned by mt_data.py:
                ((src_array, tgt_array, src_valid_len), label_array).
            device: Device to run inference on.
            max_len: Optional max decoding length. Defaults to target length.
            decode_strategy: Decoding strategy to use ("greedy" or "beam").
            beam_size: Beam size for beam search decoding (if applicable).

        Returns:
            Tuple of (predictions, attention_weights). Attention weights are
            currently not collected and are returned as an empty list.
        """
        if device is None:
            from .training.base import get_device
            device = get_device()
        was_training = self.training
        self.eval()

        (src_array, tgt_array, src_valid_len), _ = batch
        src_array = src_array.to(device)
        src_valid_len = src_valid_len.to(device)
        tgt_array = tgt_array.to(device)

        dec_X = tgt_array[:, :1] # X shape: (batch_size, 1)
        preds: list[Tensor] = []
        attn_weights: list[Tensor] = []

        decode_len = max_len if max_len is not None else tgt_array.shape[1]

        eos_token_index = self._resolve_eos_index(batch, device)

        with torch.no_grad():
            enc_outputs = self.encoder(src_array, src_valid_len)
            dec_state = self.decoder.preprocess_state(enc_outputs, src_valid_len)

            if decode_strategy == "greedy" or beam_size <= 1:
                pred_seq = self._greedy_decode(
                    dec_X,
                    dec_state,
                    decode_len,
                )
            else:
                pred_seq = self._beam_decode(
                    dec_X,
                    dec_state,
                    decode_len,
                    beam_size,
                    eos_token_index,
                    device,
                    tgt_array,
                )
            preds = [pred_seq]

        if was_training:
            self.train()

        if len(preds) == 1 and preds[0].ndim == 2:
            pred_seq = preds[0]
        else:
            pred_seq = torch.stack(preds, dim=1)
        return pred_seq, attn_weights
