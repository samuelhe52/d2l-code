import torch
from torch import nn
from utils.classfication import train, get_dataloader, test


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
        'dropout_1': 0.25,
        'dropout_2': 0.25
    }

    batch_size = 256
    num_epochs = 30
    lr = 0.25
    model = MLP(**hparams)
    save_path = 'models/mlp_no_d2l.pt'
    dataloader = get_dataloader(batch_size, data_root='data/')
    train(model, dataloader, num_epochs, lr, save_path=save_path)
    test_acc = test(model, dataloader)
    print(f'Test Accuracy: {test_acc:.4f}')
