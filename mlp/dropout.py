import torch
from torch import nn
from d2l import torch as d2l

class MLPWithDropout(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs,
                 num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.Linear(num_inputs, num_hiddens_1)
        self.lin2 = nn.Linear(num_hiddens_1, num_hiddens_2)
        self.lin3 = nn.Linear(num_hiddens_2, num_outputs)
        self.relu = nn.ReLU()
        
    def dropout_layer(self, X, dropout):
        assert 0 <= dropout <= 1
        if dropout == 1: return torch.zeros_like(X)
        mask = (torch.rand(X.shape) > dropout).float()
        return X * mask / (1 - dropout)
    
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:
            H1 = self.dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = self.dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)

hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = MLPWithDropout(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)