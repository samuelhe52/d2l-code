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
        'seq_len': 32,
        'batch_size': 1024,
        'num_hiddens': 32,
        'num_layers': 1,
        'dropout': 0.0,
        'num_epochs': 100,
        'lr': 1,
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
        save_path='./models/rnnlm_lstm.pt',
        logger=logger,
        device=torch.device('cpu')
    )
    
    # trainer = RNNTrainer(model, train_loader, val_loader, config)
    # trainer.train()
    # logger.summary()
    # logger.save()
    
    # Test generation
    model = load_model('./models/rnnlm_lstm.pt', model, device=torch.device('cpu'))
    print(model.generate(
        prefix='it is high time ',
        num_preds=50,
        vocab=data.vocab,
        device=torch.device('cpu')
    ))
