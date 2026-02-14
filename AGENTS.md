# Project Guidelines

- Primary mission: help the user understand *Dive into Deep Learning* concepts while implementing runnable code in this repo.
- Math formatting is strict: use `$$...$$` for block math and `$...$` for inline math.
- If an explanation is too long/complex for terminal chat, create a detailed note in `docs/` and point the user to it.

## Code Style

- Follow existing PyTorch-first style: type hints on public methods, concise docstrings, and shape comments where non-obvious (see `attention_transformer/bahdanau_attn.py`).
- Keep tensor layout conventions consistent:
  - Seq2Seq decoder logits: `(batch_size, vocab_size, seq_len)`.
  - Encoder/decoder token IDs are `torch.long`.
  - Valid lengths are integer tensors, typically `(batch_size,)`.
- Preserve current naming and module patterns (`hparams`, `shared`, `config`, `logger`, `trainer`) used in runnable scripts.

## Architecture

- Models are organized by topic/chapter: `linear_regression/`, `mlp/`, `rnn/`, `modern_rnns/`, `modern_cnns/`, `attention_transformer/`.
- Shared primitives live in `utils/`:
  - Attention and masking: `utils/attn.py`
  - Encoder-decoder base + decoding utilities: `utils/enc_dec.py`
  - Training framework: `utils/training/`
  - Dataset/vocab code: `utils/data/`
- Seq2Seq flow is standardized:
  1. `encoder(enc_X, *args)`
  2. `decoder.preprocess_state(enc_outputs, *args)`
  3. `decoder(dec_X, state)`
  4. Generation through `EncoderDecoder.generate(...)` with greedy/beam decode.

## Build and Test

- Environment: `conda activate d2l` (required for local torch setup in this workspace).
- Run examples directly, e.g.:
  - `python modern_rnns/seq2seq_mt.py`
  - `python attention_transformer/bahdanau_attn.py`
- Utility commands (Makefile):
  - `make dashboard` (serves dashboard on localhost)
  - `make summary model=<name>`
  - `make flops`
- Lightweight checks are script-based (`*test*.py` files), e.g. `python modern_rnns/mt_data_test.py`.

## Project Conventions

- Prefer `TrainingConfig` + `TrainingLogger` + trainer classes (`Seq2SeqTrainer`, `RNNTrainer`) over ad-hoc loops.
- Device selection pattern:
  - Training scripts may pin `torch.device('mps')`.
  - Shared trainer fallback logic is in `utils/training/base.py` (`cuda -> mps -> cpu`).
- Decoder state for beam search must be tensor/tuple/list-compatible so `_repeat_state` and `_select_state` in `utils/enc_dec.py` can reorder beams.
- Keep save/load behavior compatible with `utils/io.py` (`save_model`, `load_model`, directory checkpoint resolution).

## Integration Points

- MT datasets and loaders are centralized in `utils/data/mt_data.py` (`GerEngDataset`, `FraEngDataset`, `mt_dataloader`, `eval_translations`).
- Dashboard API/UI integration is via `dashboard_server.py` and `docs/experiment_dashboard.html`, backed by JSON logs in `logs/`.
- Tooling scripts in `tools/` import project modules by path and assume repo root importability.

## Security

- Do not hardcode secrets or tokens; this repo is local-code focused and uses file-based artifacts.
- Treat data/model artifacts as user assets: `data/`, `models/`, and `logs/` can be large and should not be deleted or overwritten casually.
- `dashboard_server.py` mutates log files via DELETE endpoints; keep path validation behavior intact when editing server logic.
