from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from utils.rnnlm import RNNLM
from utils.io import load_model
from utils.training import (
    RNNTrainer,
    TrainingConfig,
    TrainingLogger
)
from utils.data import (
    book_data_loader,
    PrideAndPrejudiceData,
    TimeMachineData,
)


class GRULM(nn.Module):
    """Adapter model for registry-driven profiling."""

    def __init__(self, vocab_size: int = 100, num_hiddens: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        rnn = nn.GRU(
            input_size=vocab_size,
            hidden_size=num_hiddens,
            batch_first=False,
        )
        self.rnnlm = RNNLM(
            vocab_size=vocab_size,
            num_hiddens=num_hiddens,
            rnn=rnn,
        )

    def forward(self, X):
        X = X.type(torch.long) % self.vocab_size
        logits, _ = self.rnnlm(X)
        return logits

if __name__ == "__main__":
    hparams = {
        'seq_len': 48,
        'batch_size': 1024,
        'num_hiddens': 256,
        'grad_clip': 1.0,
        'num_epochs': 100,
        'lr': 0.4,
        'rnn_type': 'GRU',
    }
    
    data = PrideAndPrejudiceData(seq_len=hparams['seq_len'], use_chars=True)
    train_loader = book_data_loader(
        data, batch_size=hparams['batch_size'], train=True
    )
    val_loader = book_data_loader(
        data, batch_size=hparams['batch_size'], train=False
    )
    
    rnn = nn.GRU(input_size=len(data.vocab),
                  hidden_size=hparams['num_hiddens'],
                  batch_first=False)
    
    model = RNNLM(
        vocab_size=len(data.vocab),
        num_hiddens=hparams['num_hiddens'],
        rnn=rnn
    )
    
    logger = TrainingLogger(
        log_path='logs/gru_experiment.json',
        hparams=hparams
    )
    
    config = TrainingConfig(
        num_epochs=hparams['num_epochs'],
        lr=hparams['lr'],
        loss_fn=nn.CrossEntropyLoss(),
        grad_clip=hparams['grad_clip'],
        save_path='./models/rnnlm_gru.pt',
        logger=logger,
        device=torch.device('mps'),
    )
    
    trainer = RNNTrainer(model, train_loader, val_loader, config)
    trainer.train()
    logger.summary()
    
    # Test generation
    model = load_model('./models/rnnlm_gru.pt',
                       model,
                       device=torch.device('cpu'))
    print(model.generate(
        prefix='time traveller ',
        num_preds=50,
        vocab=data.vocab,
        device=torch.device('cpu'),
        temperature=0.5
    ))
