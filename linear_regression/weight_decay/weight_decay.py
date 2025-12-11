import torch
from torch import nn
from d2l import torch as d2l

class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
    
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def l2_penalty(self, w):
        """Compute L2 penalty."""
        return (w**2).sum() / 2
    
    def loss(self, y_hat, y):
        """Compute the loss with L2 regularization."""
        return (
            super().loss(y_hat, y)
            + self.lambd * self.l2_penalty(self.w)
        )

class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            [
                {"params": self.net.weight, "weight_decay": self.wd},
                {"params": self.net.bias, "weight_decay": 0},   
            ],
            lr = self.lr,
        )
        return optimizer

data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

d2l.plt.show()
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale = 'log'
trainer.fit(model, data)
