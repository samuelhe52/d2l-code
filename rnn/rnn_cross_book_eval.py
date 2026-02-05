import torch
from torch import nn
from torch.utils.data import DataLoader

from rnn_concise import RNNLM
from utils.io import load_model
from utils.training import RNNTrainer, TrainingConfig
from utils.data import (
    TimeMachineData,
    PrideAndPrejudiceData,
    WarOfTheWorldsData,
    book_data_loader,
)


def get_device() -> torch.device:
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


def evaluate_dataset(
    model: RNNLM,
    dataset,
    batch_size: int,
    device: torch.device,
    *,
    use_val_split: bool = False,
    desc: str,
) -> float:
    """Compute perplexity on a dataset with optional train/val split."""
    if use_val_split:
        loader = book_data_loader(dataset, batch_size=batch_size, train=False)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    config = TrainingConfig(num_epochs=1, lr=0.01, loss_fn=nn.CrossEntropyLoss(), device=device)
    trainer = RNNTrainer(model, loader, None, config)
    metrics = trainer.validate()
    ppl = metrics.get('val_ppl', 0.0)
    print(f"{desc} perplexity: {ppl:.2f}")
    return ppl


def evaluate(model_path: str, seq_len: int = 32, batch_size: int = 256) -> None:
    device = get_device()
    tm_data = TimeMachineData(seq_len=seq_len, use_chars=True)
    vocab = tm_data.vocab

    model = RNNLM(vocab_size=len(vocab), num_hiddens=32)
    model = load_model(model_path, model)
    model.to(device)
    model.eval()

    print(f"Device: {device}")
    evaluate_dataset(
        model,
        tm_data,
        batch_size,
        device,
        use_val_split=True,
        desc="Time Machine (val)",
    )

    wow_data = WarOfTheWorldsData(seq_len=seq_len, use_chars=True, vocab=vocab)
    evaluate_dataset(
        model,
        wow_data,
        batch_size,
        device,
        use_val_split=False,
        desc="War of the Worlds (full)",
    )

    pap_data = PrideAndPrejudiceData(seq_len=seq_len, use_chars=True, vocab=vocab)
    evaluate_dataset(
        model,
        pap_data,
        batch_size,
        device,
        use_val_split=False,
        desc="Pride and Prejudice (full)",
    )

    prefix = "war of the worlds "
    generated = model.generate(prefix=prefix, num_preds=120, vocab=vocab, device=device)
    print("\nSample generation (prefix='war of the worlds '):")
    print(generated)


if __name__ == "__main__":
    evaluate("./models/rnnlm.pt")
