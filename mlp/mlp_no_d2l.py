import torch
from torch import nn
from utils import TrainingLogger
from utils.classfication import train, get_dataloader


class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs,
                 num_hiddens_1, num_hiddens_2,
                 dropout_1=0.0, dropout_2=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(num_inputs, num_hiddens_1),
            nn.ReLU(), nn.Dropout(dropout_1),
            nn.Linear(num_hiddens_1, num_hiddens_2),
            nn.ReLU(), nn.Dropout(dropout_2),
            nn.Linear(num_hiddens_2, num_outputs)
        )

    def forward(self, X):
        return self.net(X)

if __name__ == "__main__":
    hparams = {
        'num_inputs': 28 * 28,
        'num_outputs': 10,
        'num_hiddens_1': 512,
        'num_hiddens_2': 256,
        'dropout_1': 0.40,
        'dropout_2': 0.40,
        # 'dropout_1': 0.00,
        # 'dropout_2': 0.00,
    }

    batch_size = 256
    num_epochs = 20
    lr = 0.1
    # weight_decay = 5e-4
    weight_decay = 0.0
    
    model = MLP(**hparams)
    dataloader = get_dataloader(batch_size, data_root='data/')
    val_dataloader = get_dataloader(batch_size, train=False, data_root='data/')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Set up logger with all hyperparameters
    logger = TrainingLogger(
        log_path='logs/mlp_experiment.json',
        hparams={
            **hparams,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr': lr,
            'weight_decay': weight_decay,
        }
    )
    
    train(model, dataloader, num_epochs, lr, 
          optimizer=optimizer, logger=logger, val_dataloader=val_dataloader)
    
    logger.summary()
    logger.save()
