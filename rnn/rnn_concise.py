import torch
from torch import nn

from utils.io import load_model
from utils.rnnlm import RNNLM
from utils.training import RNNTrainer, TrainingConfig, TrainingLogger
from utils.data import (
    book_data_loader,
    TimeMachineData,
)


class RNNConciseModel(nn.Module):
    """Adapter model for registry-driven profiling."""

    def __init__(self, vocab_size: int = 100, num_hiddens: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.rnnlm = RNNLM(
            vocab_size=vocab_size,
            num_hiddens=num_hiddens,
        )

    def forward(self, X):
        X = X.type(torch.long) % self.vocab_size
        logits, _ = self.rnnlm(X)
        return logits

if __name__ == "__main__":
    hparams = {
        'seq_len': 32,
        'batch_size': 1024,
        'num_hiddens': 32,
        'num_layers': 1,
        'dropout': 0.0,
        'num_epochs': 100,
        'lr': 1,
    }
    
    data = TimeMachineData(seq_len=hparams['seq_len'], use_chars=True)
    train_loader = book_data_loader(
        data, batch_size=hparams['batch_size'], train=True
    )
    val_loader = book_data_loader(
        data, batch_size=hparams['batch_size'], train=False
    )
    model = RNNLM(
        vocab_size=len(data.vocab),
        num_hiddens=hparams['num_hiddens'],
    )

    
    logger = TrainingLogger(
        log_path='logs/rnnlm_experiment.json',
        hparams=hparams
    )
    
    config = TrainingConfig(
        num_epochs=hparams['num_epochs'],
        lr=hparams['lr'],
        loss_fn=nn.CrossEntropyLoss(),
        save_path='./models/rnnlm.pt',
        logger=logger,
        device=torch.device('cpu')
    )
    
    # trainer = RNNTrainer(model, train_loader, val_loader, config)
    # trainer.train()
    # logger.summary()
    
    # Test generation
    model = load_model('./models/rnnlm.pt', model, device=torch.device('cpu'))
    print(model.generate(
        prefix='it is high time ',
        num_preds=50,
        vocab=data.vocab,
        device=torch.device('cpu')
    ))
