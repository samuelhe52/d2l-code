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

if __name__ == "__main__":
    hparams = {
        'seq_len': 40,
        'batch_size': 1024,
        'num_hiddens': 256,
        'grad_clip': 1.0,
        'num_epochs': 100,
        'lr': 0.2,
        'rnn_type': 'LSTM',
    }
    
    data = TimeMachineData(seq_len=hparams['seq_len'], use_chars=True)
    train_loader = book_data_loader(
        data, batch_size=hparams['batch_size'], train=True
    )
    val_loader = book_data_loader(
        data, batch_size=hparams['batch_size'], train=False
    )
    
    rnn = nn.LSTM(input_size=len(data.vocab),
                  hidden_size=hparams['num_hiddens'],
                  batch_first=False)
    
    model = RNNLM(
        vocab_size=len(data.vocab),
        num_hiddens=hparams['num_hiddens'],
        rnn=rnn
    )
    
    logger = TrainingLogger(
        log_path='logs/lstm_experiment.json',
        hparams=hparams
    )
    
    config = TrainingConfig(
        num_epochs=hparams['num_epochs'],
        lr=hparams['lr'],
        loss_fn=nn.CrossEntropyLoss(),
        grad_clip=hparams['grad_clip'],
        save_path='./models/rnnlm_lstm.pt',
        logger=logger,
        device=torch.device('mps')
    )
    
    trainer = RNNTrainer(model, train_loader, val_loader, config)
    trainer.train()
    logger.summary()
    
    # Test generation
    model = load_model('./models/rnnlm_lstm.pt',
                       model,
                       device=torch.device('mps'))
    print(model.generate(
        prefix='it is high time',
        num_preds=50,
        vocab=data.vocab,
        device=torch.device('mps'),
        temperature=0.3
    ))
